#!/usr/bin/env python3
"""
BMC Multi-Round Agentic Benchmark (Research Quality)
=====================================================

Simulates three production workloads from the BMC proposal (Section 3.1),
measuring decode kernel time, CPU↔GPU transfer time, and allocation overhead
across Paged, LMCache-chunked, and BMC contiguous strategies.

Workloads:
  W1 — Multi-Turn Chat:   System prompt 2K + 10 rounds × ~300 tokens/round
  W2 — Long Document QA:  Document 8K + 5 Q&A rounds × ~200 tokens/round
  W3 — Agentic Tool-Call: 8 tool calls × variable output (50-1000 tokens)
       with CPU offload/restore between each tool call

Baselines (per the proposal Section 3.2):
  1. Paged (vLLM default):  paged_attention_v1 + N/B DMA transfers
  2. LMCache-style:         paged_attention_v1 + 2MB chunked DMA transfers
  3. BMC √N (contiguous):   SDPA + single DMA + √N reallocation

Metrics (per the proposal Section 3.3):
  - Decode throughput (tok/s)
  - DMA transfer time — offload (GPU→CPU) and restore (CPU→GPU)
  - Total allocations and reallocations
  - Total round-trip time per workload

Hardware: AMD MI300X (192 GB HBM3, PCIe Gen5)
Models: LLaMA-3-8B, Qwen2-7B (bfloat16)

Usage:
  python bench_multiround_bmc.py                           # full suite
  python bench_multiround_bmc.py --quick                   # 4 layers, quick
  python bench_multiround_bmc.py --model-config llama-3-8b # single model
  python bench_multiround_bmc.py --workloads W1 W3         # select workloads
"""

import argparse
import json
import math
import os
import random
import statistics
import time

import torch
import torch.nn.functional as F

torch.manual_seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# Model configurations (real architectures)
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "llama-3-8b": {
        "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "num_layers": 32, "hidden": 4096, "inter": 14336,
    },
    "qwen2-7b": {
        "num_heads": 28, "num_kv_heads": 4, "head_dim": 128,
        "num_layers": 28, "hidden": 3584, "inter": 18944,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Workload definitions (from BMC proposal Section 3.1)
# ═══════════════════════════════════════════════════════════════════════════

def _load_workloads(workload_dir=None):
    """Load workload definitions from JSON files."""
    if workload_dir is None:
        workload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "workloads")
    workloads = {}
    if os.path.isdir(workload_dir):
        for fname in sorted(os.listdir(workload_dir)):
            if fname.endswith(".json"):
                path = os.path.join(workload_dir, fname)
                with open(path) as f:
                    wl = json.load(f)
                wid = wl.get("id", fname.replace(".json", ""))
                workloads[wid] = wl
    if not workloads:
        # Fallback inline defaults if JSON files not found
        workloads = {
            "W1": {"name": "Multi-Turn Chat", "context_len": 2048,
                   "description": "10 chat rounds with offload",
                   "rounds": [{"decode": 300, "offload": True}] * 10},
            "W2": {"name": "Long Document QA", "context_len": 8192,
                   "description": "5 QA rounds on 8K doc",
                   "rounds": [{"decode": 200, "offload": True}] * 5},
            "W3": {"name": "Agentic Tool-Calling", "context_len": 512,
                   "description": "8 tool calls with variable output",
                   "rounds": [{"decode": d, "offload": True}
                              for d in [150, 800, 50, 400, 1000, 200, 600, 100]]},
        }
    return workloads

WORKLOADS = _load_workloads()

# ═══════════════════════════════════════════════════════════════════════════
# Transformer components
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm:
    def __init__(self, dim, dev, dt):
        self.w = torch.ones(dim, device=dev, dtype=dt); self.eps = 1e-5
    def __call__(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w, self.eps)

class FFN:
    def __init__(self, h, inter, dev, dt):
        s = h ** -0.5
        self.g = torch.empty(inter, h, device=dev, dtype=dt).uniform_(-s, s)
        self.u = torch.empty(inter, h, device=dev, dtype=dt).uniform_(-s, s)
        self.d = torch.empty(h, inter, device=dev, dtype=dt).uniform_(-s, s)
    def __call__(self, x):
        return F.linear(F.silu(F.linear(x, self.g)) * F.linear(x, self.u), self.d)

class QKVProj:
    def __init__(self, h, nh, nkv, hd, dev, dt):
        s = h ** -0.5
        self.qkv = torch.empty(nh*hd + 2*nkv*hd, h, device=dev, dtype=dt).uniform_(-s, s)
        self.o = torch.empty(h, nh*hd, device=dev, dtype=dt).uniform_(-s, s)
        self.q_dim, self.kv_dim = nh*hd, nkv*hd
    def __call__(self, x):
        qkv = F.linear(x, self.qkv)
        return qkv[...,:self.q_dim], qkv[...,self.q_dim:self.q_dim+self.kv_dim], qkv[...,self.q_dim+self.kv_dim:]

class Layer:
    def __init__(self, h, inter, nh, nkv, hd, dev, dt):
        self.n1 = RMSNorm(h, dev, dt); self.n2 = RMSNorm(h, dev, dt)
        self.qkv = QKVProj(h, nh, nkv, hd, dev, dt); self.ffn = FFN(h, inter, dev, dt)
        self.nh, self.nkv, self.hd = nh, nkv, hd


# ═══════════════════════════════════════════════════════════════════════════
# KV Cache — Contiguous (BMC)
# ═══════════════════════════════════════════════════════════════════════════

class ContiguousKV:
    def __init__(self, B, nkv, hd, cap, dt, dev):
        self.B, self.nkv, self.hd = B, nkv, hd
        self.cap, self.seq, self.dt, self.dev = cap, 0, dt, dev
        self.k = torch.zeros(B, nkv, cap, hd, dtype=dt, device=dev)
        self.v = torch.zeros(B, nkv, cap, hd, dtype=dt, device=dev)
        self.reallocs = 0

    def append(self, kn, vn):
        self.k[:,:,self.seq:self.seq+1,:] = kn
        self.v[:,:,self.seq:self.seq+1,:] = vn
        self.seq += 1

    def grow_if_needed(self):
        if self.seq + 1 > self.cap:
            sq = max(int(math.isqrt(self.cap)), 8)
            nc = self.cap + sq
            nk = torch.zeros(self.B, self.nkv, nc, self.hd, dtype=self.dt, device=self.dev)
            nv = torch.zeros(self.B, self.nkv, nc, self.hd, dtype=self.dt, device=self.dev)
            nk[:,:,:self.seq,:] = self.k[:,:,:self.seq,:]
            nv[:,:,:self.seq,:] = self.v[:,:,:self.seq,:]
            self.k, self.v, self.cap = nk, nv, nc
            self.reallocs += 1

    def offload(self):
        # BMC: 2 contiguous DMA calls (1 for K, 1 for V) — no gather needed
        ck = torch.empty(self.B, self.nkv, self.seq, self.hd,
                          dtype=self.dt, device="cpu").pin_memory()
        cv = torch.empty(self.B, self.nkv, self.seq, self.hd,
                          dtype=self.dt, device="cpu").pin_memory()
        ck.copy_(self.k[:,:,:self.seq,:], non_blocking=True)
        cv.copy_(self.v[:,:,:self.seq,:], non_blocking=True)
        torch.cuda.synchronize()
        # Free GPU memory (agent is waiting for tool response)
        self.k = None; self.v = None
        torch.cuda.empty_cache()
        return ck, cv, 2  # 2 DMA calls (K + V)

    def restore(self, ck, cv):
        # BMC: 2 contiguous DMA calls to restore
        self.seq = ck.shape[2]
        need_cap = self.seq + max(int(math.isqrt(self.seq)), 8)
        self.cap = need_cap
        self.k = torch.zeros(self.B, self.nkv, self.cap, self.hd,
                              dtype=self.dt, device=self.dev)
        self.v = torch.zeros(self.B, self.nkv, self.cap, self.hd,
                              dtype=self.dt, device=self.dev)
        self.k[:,:,:self.seq,:].copy_(ck, non_blocking=True)
        self.v[:,:,:self.seq,:].copy_(cv, non_blocking=True)
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════════
# KV Cache — Paged (vLLM format)
# ═══════════════════════════════════════════════════════════════════════════

class PagedKV:
    def __init__(self, B, nkv, hd, pool_blks, bsz, dt, dev):
        self.B, self.nkv, self.hd, self.bsz = B, nkv, hd, bsz
        self.dt, self.dev, self.seq = dt, dev, 0
        self.xp = 16 // torch.tensor([], dtype=dt).element_size()
        s = hd ** -0.5
        self.kc = torch.empty(pool_blks, nkv, hd//self.xp, bsz, self.xp,
                               dtype=dt, device=dev).uniform_(-s, s)
        self.vc = torch.empty(pool_blks, nkv, hd, bsz,
                               dtype=dt, device=dev).uniform_(-s, s)
        rng = random.Random(42)
        mb = pool_blks // max(B, 1)
        self.bt = torch.tensor([[rng.randint(0, pool_blks-1) for _ in range(mb)]
                                 for _ in range(B)], dtype=torch.int32, device=dev)
        self.sl = torch.zeros(B, dtype=torch.int32, device=dev)

    def write(self, kn, vn):
        pos = self.seq
        bi = self.bt[:, pos // self.bsz].long()
        off = pos % self.bsz
        kp = kn.view(self.B, self.nkv, self.hd // self.xp, self.xp)
        self.kc[bi, :, :, off, :] = kp
        self.vc[bi, :, :, off] = vn
        self.seq += 1; self.sl.fill_(self.seq)

    def offload(self):
        nb = math.ceil(self.seq / self.bsz)
        ids = self.bt[:, :nb].long()
        cks, cvs = [], []
        total_dma = 0
        for b in range(self.B):
            for j in range(nb):
                bid = ids[b, j].item()
                ck = torch.empty_like(self.kc[bid], device="cpu").pin_memory()
                cv = torch.empty_like(self.vc[bid], device="cpu").pin_memory()
                ck.copy_(self.kc[bid], non_blocking=True)
                cv.copy_(self.vc[bid], non_blocking=True)
                cks.append(ck); cvs.append(cv)
                total_dma += 2
        torch.cuda.synchronize()
        return cks, cvs, ids, total_dma

    def restore(self, cks, cvs, ids):
        nb = ids.shape[1]
        idx = 0; total_dma = 0
        for b in range(self.B):
            for j in range(nb):
                bid = ids[b, j].item()
                self.kc[bid].copy_(cks[idx], non_blocking=True)
                self.vc[bid].copy_(cvs[idx], non_blocking=True)
                idx += 1; total_dma += 2
        torch.cuda.synchronize()
        return total_dma


# ═══════════════════════════════════════════════════════════════════════════
# KV Cache — Paged Packed (vLLM ≥0.12 layout: all layers in one phys block)
# ═══════════════════════════════════════════════════════════════════════════

class PagedPackedKV:
    """Models vLLM 0.12+ physical block layout.

    All layers' K and V for a logical block are packed into one contiguous
    physical block (~2MB for Llama-3-8B). Offload/restore does 1 DMA per
    logical block instead of num_layers × 2 DMAs.
    """
    def __init__(self, B, nkv, hd, num_layers, bsz, dt, dev):
        self.B, self.nkv, self.hd, self.bsz = B, nkv, hd, bsz
        self.nl, self.dt, self.dev, self.seq = num_layers, dt, dev, 0
        elem = torch.tensor([], dtype=dt).element_size()
        self.block_bytes = bsz * nkv * hd * 2 * num_layers * elem
        max_blocks = max(B * 2048, 16384)
        self.pool = torch.randn(max_blocks, num_layers * 2 * bsz * nkv * hd,
                                 dtype=dt, device=dev)
        rng = random.Random(42)
        mb = max_blocks // max(B, 1)
        self.bt = torch.tensor([[rng.randint(0, max_blocks-1) for _ in range(mb)]
                                 for _ in range(B)], dtype=torch.int32, device=dev)

    def write(self, kn, vn):
        self.seq += 1

    def offload(self):
        nb = math.ceil(self.seq / self.bsz)
        ids = self.bt[:, :nb].long()
        cpu_blocks = []
        total_dma = 0
        for b in range(self.B):
            for j in range(nb):
                bid = ids[b, j].item()
                cb = torch.empty_like(self.pool[bid], device="cpu").pin_memory()
                cb.copy_(self.pool[bid], non_blocking=True)
                cpu_blocks.append(cb)
                total_dma += 1
        torch.cuda.synchronize()
        return cpu_blocks, ids, total_dma

    def restore(self, cpu_blocks, ids):
        nb = ids.shape[1]; idx = 0; total_dma = 0
        for b in range(self.B):
            for j in range(nb):
                bid = ids[b, j].item()
                self.pool[bid].copy_(cpu_blocks[idx], non_blocking=True)
                idx += 1; total_dma += 1
        torch.cuda.synchronize()
        return total_dma


# ═══════════════════════════════════════════════════════════════════════════
# LMCache-style offload (2MB chunks from gathered contiguous)
# ═══════════════════════════════════════════════════════════════════════════

class LMCacheKV(ContiguousKV):
    """Same contiguous GPU buffer as BMC, but offload/restore in 2MB chunks.

    This models LMCache v0.12's approach: paged GPU KV is gathered into
    contiguous CPU buffers using fixed-size 2MB DMA chunks. On restore,
    the chunks are copied back and scattered into GPU pages.
    """
    CHUNK = 2 * 1024 * 1024  # 2 MB per LMCache default

    def offload(self):
        flat = torch.cat([self.k[:,:,:self.seq,:].reshape(-1),
                           self.v[:,:,:self.seq,:].reshape(-1)])
        nbytes = flat.numel() * flat.element_size()
        nchunks = math.ceil(nbytes / self.CHUNK)
        chunk_elems = self.CHUNK // flat.element_size()
        cpu_chunks = []
        for i in range(nchunks):
            s = i * chunk_elems
            e = min(s + chunk_elems, flat.numel())
            cc = torch.empty(e - s, dtype=flat.dtype, device="cpu").pin_memory()
            cc.copy_(flat[s:e], non_blocking=True)
            cpu_chunks.append(cc)
        torch.cuda.synchronize()
        self.k = None; self.v = None
        torch.cuda.empty_cache()
        return cpu_chunks, nchunks

    def restore(self, cpu_chunks):
        flat = torch.cat(cpu_chunks).to(self.dev, non_blocking=True)
        torch.cuda.synchronize()
        half = flat.numel() // 2
        kf = flat[:half].reshape(self.B, self.nkv, self.seq, self.hd)
        vf = flat[half:].reshape(self.B, self.nkv, self.seq, self.hd)
        need_cap = self.seq + max(int(math.isqrt(self.seq)), 8)
        self.cap = need_cap
        self.k = torch.zeros(self.B, self.nkv, self.cap, self.hd,
                              dtype=self.dt, device=self.dev)
        self.v = torch.zeros(self.B, self.nkv, self.cap, self.hd,
                              dtype=self.dt, device=self.dev)
        self.k[:,:,:self.seq,:] = kf
        self.v[:,:,:self.seq,:] = vf
        del flat, kf, vf


# ═══════════════════════════════════════════════════════════════════════════
# Attention kernels
# ═══════════════════════════════════════════════════════════════════════════

_GQA = None
def _probe_gqa(nh, nkv, hd, dt, dev):
    global _GQA
    if _GQA: return
    if nh == nkv: _GQA = "mha"; return
    try:
        F.scaled_dot_product_attention(
            torch.randn(1,nh,1,hd,dtype=dt,device=dev),
            torch.randn(1,nkv,4,hd,dtype=dt,device=dev),
            torch.randn(1,nkv,4,hd,dtype=dt,device=dev),
            enable_gqa=True); _GQA = "native"
    except TypeError: _GQA = "bcast"

def sdpa(q, k, v, nh, nkv, scale):
    if _GQA in ("mha", "native"):
        kw = {"enable_gqa": True} if _GQA == "native" else {}
        return F.scaled_dot_product_attention(q, k, v, scale=scale,
                                              is_causal=False, **kw)
    B, _, S, D = q.shape; g = nh // nkv
    return F.scaled_dot_product_attention(
        q.view(B, nkv, g, S, D), k.unsqueeze(2), v.unsqueeze(2),
        scale=scale, is_causal=False).view(B, nh, S, D)

_PA = None; _KS = None
def probe_pa(dt):
    global _PA, _KS
    if _PA is not None: return _PA
    try:
        from vllm._custom_ops import paged_attention_v1
        h, d, b = 1, 128, 16; x = 16 // torch.tensor([],dtype=dt).element_size()
        q = torch.randn(1,h,d,dtype=dt,device="cuda")
        k = torch.empty(4,h,d//x,b,x,dtype=dt,device="cuda").uniform_(-0.1,0.1)
        v = torch.empty(4,h,d,b,dtype=dt,device="cuda").uniform_(-0.1,0.1)
        bt = torch.tensor([[0,1]],dtype=torch.int32,device="cuda")
        sl = torch.tensor([2*b],dtype=torch.int32,device="cuda")
        o = torch.empty_like(q); ks = torch.tensor(1.0,dtype=torch.float32,device="cuda")
        paged_attention_v1(o,q,k,v,h,1/math.sqrt(d),bt,sl,b,2*b,None,"auto",ks,ks)
        torch.cuda.synchronize(); _PA = True; _KS = ks
    except Exception: _PA = False
    return _PA

# ═══════════════════════════════════════════════════════════════════════════
# Decode step
# ═══════════════════════════════════════════════════════════════════════════

def step_bmc(h, layers, kvs, sc):
    B = h.shape[0]
    for i, L in enumerate(layers):
        r = h; x = L.n1(h); q, kn, vn = L.qkv(x)
        q = q.view(B,L.nh,1,L.hd); kn = kn.view(B,L.nkv,1,L.hd); vn = vn.view(B,L.nkv,1,L.hd)
        kvs[i].grow_if_needed(); kvs[i].append(kn, vn)
        kf, vf = kvs[i].k, kvs[i].v
        o = sdpa(q, kf, vf, L.nh, L.nkv, sc).view(B,1,-1)
        h = F.linear(o, L.qkv.o) + r; r = h; h = L.ffn(L.n2(h)) + r
    return h

def step_paged(h, layers, kvs, sc, bsz, use_pa):
    B = h.shape[0]
    for i, L in enumerate(layers):
        r = h; x = L.n1(h); q, kn, vn = L.qkv(x)
        kf = kn.view(B,L.nkv,L.hd); vf = vn.view(B,L.nkv,L.hd)
        kvs[i].write(kf, vf)
        q3 = q.view(B,L.nh,L.hd)
        if use_pa:
            from vllm._custom_ops import paged_attention_v1
            o3 = torch.empty_like(q3)
            paged_attention_v1(o3, q3, kvs[i].kc, kvs[i].vc, L.nkv, sc,
                               kvs[i].bt, kvs[i].sl, bsz,
                               int(kvs[i].sl.max().item()), None, "auto", _KS, _KS)
        else:
            ms = int(kvs[i].sl.max().item()); mb = math.ceil(ms/bsz)
            idx = kvs[i].bt[:,:mb].long()
            kg = kvs[i].kc[idx].permute(0,2,1,4,3,5).contiguous().reshape(B,L.nkv,mb*bsz,L.hd)
            vg = kvs[i].vc[idx].permute(0,2,1,4,3).contiguous().reshape(B,L.nkv,mb*bsz,L.hd)
            o3 = sdpa(q.view(B,L.nh,1,L.hd), kg, vg, L.nh, L.nkv, sc).squeeze(2)
        h = F.linear(o3.view(B,1,-1), L.qkv.o) + r; r = h; h = L.ffn(L.n2(h)) + r
    return h


# ═══════════════════════════════════════════════════════════════════════════
# Timer
# ═══════════════════════════════════════════════════════════════════════════

class T:
    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
    def go(self): self.s.record()
    def stop(self): self.e.record(); torch.cuda.synchronize(); return self.s.elapsed_time(self.e)


# ═══════════════════════════════════════════════════════════════════════════
# Run one workload with one strategy
# ═══════════════════════════════════════════════════════════════════════════

def run_workload(strategy, workload, layers, cfg, batch, bsz, dt, dev, use_pa):
    nl = len(layers); nkv = cfg["num_kv_heads"]; hd = cfg["head_dim"]
    sc = hd ** -0.5; ctx = workload["context_len"]
    timer = T()

    # Initialize KV caches
    if strategy in ("paged", "paged-packed"):
        total_seq = ctx + sum(r["decode"] for r in workload["rounds"])
        pool = max(math.ceil(total_seq / bsz) * batch * 4, 4096)
        kvs = [PagedKV(batch, nkv, hd, pool, bsz, dt, dev) for _ in range(nl)]
        for kv in kvs: kv.seq = ctx; kv.sl.fill_(ctx)
        if strategy == "paged-packed":
            # Additional packed transfer buffer (models v0.12+ layout)
            packed_buf = PagedPackedKV(batch, nkv, hd, nl, bsz, dt, dev)
            packed_buf.seq = ctx
    elif strategy == "lmcache":
        sq = max(int(math.isqrt(ctx)), 8)
        kvs = [LMCacheKV(batch, nkv, hd, ctx + sq, dt, dev) for _ in range(nl)]
        for kv in kvs: kv.seq = ctx
    else:
        sq = max(int(math.isqrt(ctx)), 8)
        kvs = [ContiguousKV(batch, nkv, hd, ctx + sq, dt, dev) for _ in range(nl)]
        for kv in kvs: kv.seq = ctx

    hid = torch.randn(batch, 1, cfg["hidden"], dtype=dt, device=dev)

    # Use paged decode kernel for paged/paged-packed, BMC SDPA for bmc/lmcache
    use_paged_decode = strategy in ("paged", "paged-packed")

    # Warmup
    for _ in range(2):
        if use_paged_decode:
            step_paged(hid, layers, kvs, sc, bsz, use_pa)
        else:
            step_bmc(hid, layers, kvs, sc)
    torch.cuda.synchronize()
    if use_paged_decode:
        for kv in kvs: kv.seq = ctx; kv.sl.fill_(ctx)
    else:
        for kv in kvs: kv.seq = ctx

    results = []
    cpu_states = [None] * nl

    for ri, rd in enumerate(workload["rounds"]):
        dec = rd["decode"]; do_offload = rd["offload"]

        # Restore
        restore_ms = 0.0; restore_dma = 0
        if cpu_states[0] is not None:
            timer.go()
            if strategy == "paged":
                for i in range(nl):
                    d = kvs[i].restore(*cpu_states[i]); restore_dma += d
                    cpu_states[i] = None
            elif strategy == "paged-packed":
                # v0.12+: 1 DMA per physical block (all layers packed, ~2MB each)
                d = packed_buf.restore(*cpu_states[0]); restore_dma += d
                cpu_states[0] = None
            elif strategy == "lmcache":
                for i in range(nl):
                    kvs[i].restore(cpu_states[i]); cpu_states[i] = None
                elem_sz = torch.tensor([], dtype=dt).element_size()
                restore_dma = sum(math.ceil(kvs[0].seq * nkv * hd * 2 * 2 *
                                  elem_sz / LMCacheKV.CHUNK) for _ in range(nl))
            else:  # bmc
                for i in range(nl):
                    kvs[i].restore(*cpu_states[i]); cpu_states[i] = None
                restore_dma = nl
            restore_ms = timer.stop()

        # Decode
        timer.go()
        for _ in range(dec):
            if strategy == "paged":
                step_paged(hid, layers, kvs, sc, bsz, use_pa)
            elif strategy == "paged-packed":
                step_paged(hid, layers, kvs, sc, bsz, use_pa)
                packed_buf.seq = kvs[0].seq
            else:  # bmc or lmcache
                step_bmc(hid, layers, kvs, sc)
        decode_ms = timer.stop()

        # Offload
        offload_ms = 0.0; offload_dma = 0
        if do_offload:
            timer.go()
            if strategy == "paged":
                for i in range(nl):
                    state = kvs[i].offload()
                    cpu_states[i] = (state[0], state[1], state[2])
                    offload_dma += state[3]
            elif strategy == "paged-packed":
                # v0.12+: 1 DMA per physical block (all layers packed, ~2MB each)
                packed_buf.seq = kvs[0].seq
                state = packed_buf.offload()
                cpu_states[0] = (state[0], state[1])
                offload_dma += state[2]
            elif strategy == "lmcache":
                for i in range(nl):
                    chunks, nc = kvs[i].offload()
                    cpu_states[i] = chunks
                    offload_dma += nc
            else:  # bmc
                for i in range(nl):
                    ck, cv, nd = kvs[i].offload()
                    cpu_states[i] = (ck, cv)
                    offload_dma += nd
            offload_ms = timer.stop()

        seq_now = kvs[0].seq if hasattr(kvs[0], 'seq') else ctx
        reallocs = sum(getattr(kv, 'reallocs', 0) for kv in kvs) // nl if strategy not in ("paged", "paged-packed") else 0

        results.append({
            "round": ri + 1, "decode_tokens": dec, "seq_len": seq_now,
            "decode_ms": decode_ms, "offload_ms": offload_ms,
            "restore_ms": restore_ms, "offload_dma": offload_dma,
            "restore_dma": restore_dma, "reallocs": reallocs,
            "tok_per_s": (dec * batch) / (decode_ms / 1000) if decode_ms > 0 else 0,
        })

    del kvs; torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="BMC Multi-Round Agentic Benchmark")
    p.add_argument("--model-configs", nargs="+",
                   choices=list(MODEL_CONFIGS.keys()), default=["qwen2-7b"])
    p.add_argument("--workloads", nargs="+",
                   choices=list(WORKLOADS.keys()), default=["W1", "W2", "W3"])
    p.add_argument("--strategies", nargs="+",
                   choices=["paged", "paged-packed", "lmcache", "bmc"],
                   default=["paged", "paged-packed", "lmcache", "bmc"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--num-runs", type=int, default=1,
                   help="Repetitions for statistical significance")
    p.add_argument("--output-dir", default="./bmc_results")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.num_layers = 4
        for wid in WORKLOADS:
            WORKLOADS[wid]["rounds"] = WORKLOADS[wid]["rounds"][:3]

    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    dev = "cuda"
    if not torch.cuda.is_available():
        print("No GPU"); return

    use_pa = probe_pa(dt)
    props = torch.cuda.get_device_properties(0)

    print("=" * 120)
    print("  BMC Multi-Round Agentic Benchmark (Research Quality)")
    print("=" * 120)
    print(f"  GPU:          {props.name} ({props.total_memory/1e9:.0f} GB)")
    print(f"  Paged kernel: {'vLLM paged_attention_v1' if use_pa else 'gather+SDPA'}")
    print(f"  BMC kernel:   F.scaled_dot_product_attention")
    print(f"  Strategies:   {args.strategies}")
    print(f"  Workloads:    {args.workloads}")
    print("=" * 120)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for model_name in args.model_configs:
        cfg = dict(MODEL_CONFIGS[model_name])
        if args.num_layers > 0: cfg["num_layers"] = args.num_layers
        _probe_gqa(cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"], dt, dev)

        print(f"\n  Building {cfg['num_layers']} layers for {model_name}...")
        layers = [Layer(cfg["hidden"], cfg["inter"], cfg["num_heads"],
                         cfg["num_kv_heads"], cfg["head_dim"], dev, dt)
                  for _ in range(cfg["num_layers"])]
        torch.cuda.synchronize()

        for wid in args.workloads:
            wl = WORKLOADS[wid]
            print(f"\n  {'━' * 115}")
            print(f"  {wid}: {wl['name']} — {wl['description']}")
            print(f"  Context: {wl['context_len']}, Rounds: {len(wl['rounds'])}, "
                  f"Total decode: {sum(r['decode'] for r in wl['rounds'])}")
            print(f"  {'━' * 115}")

            strat_totals = {}

            for strat in args.strategies:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                print(f"\n  ── {strat.upper()} ──")
                results = run_workload(strat, wl, layers, cfg, args.batch_size,
                                        args.block_size, dt, dev, use_pa)

                print(f"  {'Rnd':>3} {'Dec':>5} {'SeqL':>6} │"
                      f" {'Decode':>9} {'Offload':>9} {'Restore':>9} {'Total':>9} │"
                      f" {'Tok/s':>7} {'DMA_off':>7} {'DMA_res':>7} {'#RA':>4}")
                print(f"  {'─' * 100}")

                for r in results:
                    tot = r["decode_ms"] + r["offload_ms"] + r["restore_ms"]
                    print(f"  {r['round']:>3} {r['decode_tokens']:>5} {r['seq_len']:>6} │"
                          f" {r['decode_ms']:>8.1f}ms {r['offload_ms']:>8.2f}ms"
                          f" {r['restore_ms']:>8.2f}ms {tot:>8.1f}ms │"
                          f" {r['tok_per_s']:>7.0f} {r['offload_dma']:>7}"
                          f" {r['restore_dma']:>7} {r['reallocs']:>4}")

                td = sum(r["decode_ms"] for r in results)
                to = sum(r["offload_ms"] for r in results)
                tr = sum(r["restore_ms"] for r in results)
                tt = td + to + tr
                strat_totals[strat] = {"decode": td, "offload": to, "restore": tr, "total": tt}

                print(f"  {'─' * 100}")
                print(f"  TOTAL: decode={td:.1f}ms  offload={to:.1f}ms"
                      f"  restore={tr:.1f}ms  total={tt:.1f}ms")

            # Comparison table
            print(f"\n  {'═' * 80}")
            print(f"  {wid} COMPARISON — {model_name}")
            print(f"  {'═' * 80}")
            print(f"  {'Strategy':>10} {'Decode':>10} {'Offload':>10}"
                  f" {'Restore':>10} {'Total':>10} {'Speedup':>9}")
            print(f"  {'─' * 65}")

            base = strat_totals.get("paged", {}).get("total", 1)
            for s in args.strategies:
                t = strat_totals[s]
                spd = base / t["total"] if t["total"] > 0 else 0
                print(f"  {s:>10} {t['decode']:>9.1f}ms {t['offload']:>9.1f}ms"
                      f" {t['restore']:>9.1f}ms {t['total']:>9.1f}ms"
                      f" {spd:>8.2f}×")

            all_results[f"{model_name}_{wid}"] = strat_totals

        del layers; torch.cuda.empty_cache()

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.output_dir, f"multiround_{ts}.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {path}")

    # LaTeX
    print(f"\n  LaTeX table:")
    for key, totals in all_results.items():
        base_t = totals.get("paged", {}).get("total", 1)
        print(f"  % {key}")
        for s, t in totals.items():
            spd = base_t / t["total"] if t["total"] > 0 else 0
            label = {"paged": "Paged (v0.11)", "paged-packed": "Paged (v0.12+)",
                     "lmcache": "LMCache", "bmc": r"BMC $\sqrt{N}$"}.get(s, s)
            print(f"  {label} & {t['decode']:.0f} & {t['offload']:.0f} &"
                  f" {t['restore']:.0f} & {t['total']:.0f} &"
                  f" {spd:.2f}$\\times$ \\\\")


if __name__ == "__main__":
    main()
