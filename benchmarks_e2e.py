#!/usr/bin/env python3
"""
End-to-End Model Benchmark: Paged vs BMC 
====================================================

Full transformer decode loop: RMSNorm → Attention → Residual → RMSNorm → FFN → Residual

Paged path:
  - vLLM's paged_attention_v1 fused kernel (no gather, reads scattered block.s directly)
  - vLLM-format packed KV cache: K=[N,H,D//x,B,x], V=[N,H,D,B]
  - New K/V tokens written into pool each decode step (symmetric with BMC)
  - Falls back to gather+SDPA only if vLLM ops are unavailable

BMC path:
  - ContiguousKVCache with kv_full() — wasteful compute on full capacity tensor
  - grow() every sqrt(N) steps
  - F.scaled_dot_product_attention
  - New K/V tokens appended each decode step

Usage:
  python bench_e2e_model_bmc.py --model-config llama-3-8b --mode both
  python bench_e2e_model_bmc.py --model-config llama-3-8b --mode both --batch-size 8
  python bench_e2e_model_bmc.py --model-config llama-3-8b --num-layers 4 --mode both
"""

import argparse
import math
import random
import statistics
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

torch.manual_seed(42)
random.seed(42)

MODEL_CONFIGS = {
    "llama-2-7b":  {"num_heads": 32, "num_kv_heads": 32, "head_dim": 128,
                     "num_layers": 32, "hidden": 4096, "inter": 11008},
    "llama-3-8b":  {"num_heads": 32, "num_kv_heads": 8,  "head_dim": 128,
                     "num_layers": 32, "hidden": 4096, "inter": 14336},
    "llama-3-70b": {"num_heads": 64, "num_kv_heads": 8,  "head_dim": 128,
                     "num_layers": 80, "hidden": 8192, "inter": 28672},
    "yi-6b":       {"num_heads": 32, "num_kv_heads": 4,  "head_dim": 128,
                     "num_layers": 32, "hidden": 4096, "inter": 11008},
    "qwen2-7b":    {"num_heads": 28, "num_kv_heads": 4,  "head_dim": 128,
                     "num_layers": 28, "hidden": 3584, "inter": 18944},
}

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}


# ═══════════════════════════════════════════════════════════════════════
#  Probe vLLM paged_attention_v1  [from microbenchmark]
# ═══════════════════════════════════════════════════════════════════════

_PA_AVAILABLE = None
_KV_SCALE = None


def probe_paged_attention(dtype):
    global _PA_AVAILABLE, _KV_SCALE
    if _PA_AVAILABLE is not None:
        return _PA_AVAILABLE

    print("  ── Paged attention probe ──")

    # Step 1: Can we import vllm at all?
    try:
        import vllm
        print(f"    vLLM version:  {vllm.__version__}")
        print(f"    vLLM path:     {vllm.__file__}")
    except ImportError:
        print("    vLLM: NOT INSTALLED")
        _PA_AVAILABLE = False
        print("  ★ RESULT: gather+SDPA fallback (no vLLM)")
        return _PA_AVAILABLE

    # Step 2: Can we import the custom ops?
    try:
        from vllm._custom_ops import paged_attention_v1
        print("    paged_attention_v1: imported OK")
    except ImportError as e:
        print(f"    paged_attention_v1: IMPORT FAILED ({e})")
        _PA_AVAILABLE = False
        print("  ★ RESULT: gather+SDPA fallback (import failed)")
        return _PA_AVAILABLE

    # Step 3: Smoke test — actually run the kernel
    try:
        h, d, blk = 1, 128, 16
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        q = torch.randn(1, h, d, dtype=dtype, device="cuda")
        k = torch.empty(4, h, d // x, blk, x, dtype=dtype, device="cuda").uniform_(-0.1, 0.1)
        v = torch.empty(4, h, d, blk, dtype=dtype, device="cuda").uniform_(-0.1, 0.1)
        bt = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        sl = torch.tensor([2 * blk], dtype=torch.int32, device="cuda")
        out = torch.empty_like(q)
        ks = vs = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        paged_attention_v1(out, q, k, v, h, 1.0 / math.sqrt(d),
                           bt, sl, blk, 2 * blk, None, "auto", ks, vs)
        torch.cuda.synchronize()
        _PA_AVAILABLE = True
        _KV_SCALE = ks
        print("    smoke test:    PASSED")
        print("  ★ RESULT: vLLM paged_attention_v1 (fused kernel)")
    except Exception as e:
        _PA_AVAILABLE = False
        print(f"    smoke test:    FAILED ({e})")
        print("  ★ RESULT: gather+SDPA fallback (kernel failed)")

    return _PA_AVAILABLE


# ═══════════════════════════════════════════════════════════════════════
#  Transformer Layer
# ═══════════════════════════════════════════════════════════════════════

class RMSNorm:
    def __init__(self, dim, device, dtype):
        self.weight = torch.ones(dim, device=device, dtype=dtype)
        self.eps = 1e-5
    def __call__(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class FFN:
    def __init__(self, hidden, inter, device, dtype):
        scale = 1.0 / math.sqrt(hidden)
        self.gate = torch.empty(inter, hidden, device=device, dtype=dtype).uniform_(-scale, scale)
        self.up   = torch.empty(inter, hidden, device=device, dtype=dtype).uniform_(-scale, scale)
        self.down = torch.empty(hidden, inter, device=device, dtype=dtype).uniform_(-scale, scale)
    def __call__(self, x):
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)


class QKVProj:
    def __init__(self, hidden, num_heads, num_kv_heads, head_dim, device, dtype):
        scale = 1.0 / math.sqrt(hidden)
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.qkv = torch.empty(q_dim + 2 * kv_dim, hidden, device=device, dtype=dtype).uniform_(-scale, scale)
        self.o = torch.empty(hidden, q_dim, device=device, dtype=dtype).uniform_(-scale, scale)
        self.q_dim = q_dim
        self.kv_dim = kv_dim
    def __call__(self, x):
        qkv = F.linear(x, self.qkv)
        return (qkv[..., :self.q_dim],
                qkv[..., self.q_dim:self.q_dim + self.kv_dim],
                qkv[..., self.q_dim + self.kv_dim:])


class TransformerLayer:
    def __init__(self, hidden, inter, num_heads, num_kv_heads, head_dim, device, dtype):
        self.norm1 = RMSNorm(hidden, device, dtype)
        self.norm2 = RMSNorm(hidden, device, dtype)
        self.qkv_proj = QKVProj(hidden, num_heads, num_kv_heads, head_dim, device, dtype)
        self.ffn = FFN(hidden, inter, device, dtype)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


# ═══════════════════════════════════════════════════════════════════════
#  KV Cache — BMC (Contiguous)
# ═══════════════════════════════════════════════════════════════════════

class ContiguousKVCache:
    """[B, num_kv_heads, capacity, head_dim] — SDPA-native layout."""

    def __init__(self, batch, num_kv_heads, head_dim, capacity, dtype, device):
        self.batch = batch
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.seq_len = 0
        self.dtype = dtype
        self.device = device
        self.k = torch.zeros(batch, num_kv_heads, capacity, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros(batch, num_kv_heads, capacity, head_dim, dtype=dtype, device=device)

    def append_kv(self, k_new, v_new):
        n = k_new.shape[2]
        self.k[:, :, self.seq_len:self.seq_len + n, :] = k_new
        self.v[:, :, self.seq_len:self.seq_len + n, :] = v_new
        self.seq_len += n

    def kv_full(self):
        return self.k, self.v

    def grow(self, new_cap):
        new_k = torch.zeros(self.batch, self.num_kv_heads, new_cap, self.head_dim,
                            dtype=self.dtype, device=self.device)
        new_v = torch.zeros(self.batch, self.num_kv_heads, new_cap, self.head_dim,
                            dtype=self.dtype, device=self.device)
        new_k[:, :, :self.seq_len, :].copy_(self.k[:, :, :self.seq_len, :])
        new_v[:, :, :self.seq_len, :].copy_(self.v[:, :, :self.seq_len, :])
        del self.k, self.v
        self.k, self.v = new_k, new_v
        self.capacity = new_cap


# ═══════════════════════════════════════════════════════════════════════
#  KV Cache — Paged (vLLM packed format)
# ═══════════════════════════════════════════════════════════════════════

class PagedKVCache:
    """
    vLLM-format paged KV cache with token-level writes.
    K: [num_blocks, num_kv_heads, head_dim//x, block_size, x]
    V: [num_blocks, num_kv_heads, head_dim, block_size]
    """

    def __init__(self, batch, num_kv_heads, head_dim, pool_blocks, block_size, dtype, device):
        self.batch = batch
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.pool_blocks = pool_blocks
        self.dtype = dtype
        self.device = device
        self.seq_len = 0

        self.x_pack = 16 // torch.tensor([], dtype=dtype).element_size()

        scale = head_dim ** -0.5
        self.k_cache = torch.empty(
            pool_blocks, num_kv_heads, head_dim // self.x_pack, block_size, self.x_pack,
            dtype=dtype, device=device).uniform_(-scale, scale)
        self.v_cache = torch.empty(
            pool_blocks, num_kv_heads, head_dim, block_size,
            dtype=dtype, device=device).uniform_(-scale, scale)

        self._rng = random.Random(42)
        max_blocks_per_seq = pool_blocks // max(batch, 1)
        bt = [[self._rng.randint(0, pool_blocks - 1) for _ in range(max_blocks_per_seq)]
              for _ in range(batch)]
        self.block_table = torch.tensor(bt, dtype=torch.int32, device=device)
        self.seq_lens = torch.zeros(batch, dtype=torch.int32, device=device)

    def write_kv(self, k_new, v_new):
        """Write one new K/V token per batch element into the paged pool.
        k_new: [B, num_kv_heads, head_dim], v_new: same.
        Fully vectorized — no Python loop over batch."""
        pos = self.seq_len
        block_indices = self.block_table[:, pos // self.block_size].long()  # [B]
        offset = pos % self.block_size
        k_packed = k_new.view(self.batch, self.num_kv_heads,
                              self.head_dim // self.x_pack, self.x_pack)
        self.k_cache[block_indices, :, :, offset, :] = k_packed
        self.v_cache[block_indices, :, :, offset] = v_new
        self.seq_len += 1
        self.seq_lens.fill_(self.seq_len)

    def get_attention_args(self):
        """Return (k_cache, v_cache, block_table, seq_lens) for paged_attention_v1."""
        return self.k_cache, self.v_cache, self.block_table, self.seq_lens


# ═══════════════════════════════════════════════════════════════════════
#  Attention functions
# ═══════════════════════════════════════════════════════════════════════

_GQA_MODE = None

def _probe_gqa(num_heads, num_kv_heads, head_dim, dtype, device):
    global _GQA_MODE
    if num_heads == num_kv_heads:
        _GQA_MODE = "mha"; return
    try:
        q = torch.randn(1, num_heads, 1, head_dim, dtype=dtype, device=device)
        k = torch.randn(1, num_kv_heads, 4, head_dim, dtype=dtype, device=device)
        F.scaled_dot_product_attention(q, k, k, scale=1.0, is_causal=False, enable_gqa=True)
        _GQA_MODE = "native_gqa"
    except TypeError:
        _GQA_MODE = "broadcast"


def bmc_attention(q, k_full, v_full, num_heads, num_kv_heads, scale):
    """SDPA on contiguous KV — the BMC kernel."""
    if _GQA_MODE == "mha" or _GQA_MODE == "native_gqa":
        kw = {"enable_gqa": True} if _GQA_MODE == "native_gqa" else {}
        return F.scaled_dot_product_attention(q, k_full, v_full, scale=scale, is_causal=False, **kw)
    B, _, Sq, D = q.shape
    gqa = num_heads // num_kv_heads
    q2 = q.view(B, num_kv_heads, gqa, Sq, D)
    out = F.scaled_dot_product_attention(q2, k_full.unsqueeze(2), v_full.unsqueeze(2),
                                         scale=scale, is_causal=False)
    return out.view(B, num_heads, Sq, D)


def paged_attention(query_3d, k_cache, v_cache, block_table, seq_lens,
                    num_kv_heads, scale, block_size):
    """vLLM paged_attention_v1 fused kernel. query: [B, num_heads, head_dim]."""
    from vllm._custom_ops import paged_attention_v1
    out = torch.empty_like(query_3d)
    paged_attention_v1(out, query_3d, k_cache, v_cache,
                       num_kv_heads, scale, block_table, seq_lens,
                       block_size, int(seq_lens.max().item()),
                       None, "auto", _KV_SCALE, _KV_SCALE)
    return out


def paged_attention_gather_fallback(query_3d, k_cache, v_cache, block_table, seq_lens,
                                     num_heads, num_kv_heads, head_dim, scale, block_size):
    """Gather+SDPA fallback when vLLM ops are unavailable."""
    batch = query_3d.shape[0]
    max_s = int(seq_lens.max().item())
    max_blk = math.ceil(max_s / block_size)
    idx = block_table[:, :max_blk].long()

    k_g = k_cache[idx]
    x_pack = k_cache.shape[4]
    k_g = k_g.permute(0, 2, 1, 4, 3, 5).contiguous()
    k_g = k_g.view(batch, num_kv_heads, max_blk * block_size, head_dim)

    v_g = v_cache[idx]
    v_g = v_g.permute(0, 2, 1, 4, 3).contiguous()
    v_g = v_g.view(batch, num_kv_heads, max_blk * block_size, head_dim)

    q = query_3d.unsqueeze(2)
    return bmc_attention(q, k_g, v_g, num_heads, num_kv_heads, scale).squeeze(2)


# ═══════════════════════════════════════════════════════════════════════
#  Decode steps
# ═══════════════════════════════════════════════════════════════════════

def decode_step_bmc(hidden, layers, kv_caches, scale):
    B = hidden.shape[0]
    for i, layer in enumerate(layers):
        residual = hidden
        x = layer.norm1(hidden)
        q, k_new, v_new = layer.qkv_proj(x)
        q = q.view(B, layer.num_heads, 1, layer.head_dim)
        k_new = k_new.view(B, layer.num_kv_heads, 1, layer.head_dim)
        v_new = v_new.view(B, layer.num_kv_heads, 1, layer.head_dim)

        kv_caches[i].append_kv(k_new, v_new)
        k_full, v_full = kv_caches[i].kv_full()

        attn_out = bmc_attention(q, k_full, v_full, layer.num_heads, layer.num_kv_heads, scale)
        attn_out = attn_out.view(B, 1, -1)
        hidden = F.linear(attn_out, layer.qkv_proj.o) + residual

        residual = hidden
        hidden = layer.ffn(layer.norm2(hidden)) + residual
    return hidden


def decode_step_paged(hidden, layers, kv_caches, scale, block_size, use_vllm_kernel):
    B = hidden.shape[0]
    for i, layer in enumerate(layers):
        residual = hidden
        x = layer.norm1(hidden)
        q, k_new, v_new = layer.qkv_proj(x)

        k_flat = k_new.view(B, layer.num_kv_heads, layer.head_dim)
        v_flat = v_new.view(B, layer.num_kv_heads, layer.head_dim)
        kv_caches[i].write_kv(k_flat, v_flat)

        q_3d = q.view(B, layer.num_heads, layer.head_dim)
        k_c, v_c, bt, sl = kv_caches[i].get_attention_args()

        if use_vllm_kernel:
            attn_out = paged_attention(q_3d, k_c, v_c, bt, sl,
                                       layer.num_kv_heads, scale, block_size)
        else:
            attn_out = paged_attention_gather_fallback(
                q_3d, k_c, v_c, bt, sl,
                layer.num_heads, layer.num_kv_heads, layer.head_dim, scale, block_size)

        attn_out = attn_out.view(B, 1, -1)
        hidden = F.linear(attn_out, layer.qkv_proj.o) + residual

        residual = hidden
        hidden = layer.ffn(layer.norm2(hidden)) + residual
    return hidden


# ═══════════════════════════════════════════════════════════════════════
#  Run benchmark
# ═══════════════════════════════════════════════════════════════════════

class CudaTimer:
    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self):
        self.e.record(); torch.cuda.synchronize()
        return self.s.elapsed_time(self.e)


def run_bmc(layers, cfg, batch, context_len, decode_len, dtype, device):
    scale = cfg["head_dim"] ** -0.5
    grow_chunk = max(1, int(math.sqrt(context_len + decode_len)))
    initial_cap = context_len + grow_chunk

    kv_caches = [ContiguousKVCache(batch, cfg["num_kv_heads"], cfg["head_dim"],
                                    initial_cap, dtype, device)
                 for _ in range(len(layers))]
    for kv in kv_caches:
        kv.seq_len = context_len

    hidden = torch.randn(batch, 1, cfg["hidden"], dtype=dtype, device=device)
    timer = CudaTimer()

    for _ in range(3):
        decode_step_bmc(hidden, layers, kv_caches, scale)
    torch.cuda.synchronize()
    for kv in kv_caches:
        kv.seq_len = context_len

    realloc_count = 0
    timer.start()
    for step in range(1, decode_len + 1):
        for kv in kv_caches:
            if kv.seq_len + 1 > kv.capacity:
                kv.grow(kv.capacity + grow_chunk)
                realloc_count += 1
        decode_step_bmc(hidden, layers, kv_caches, scale)
    total_ms = timer.stop()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del kv_caches; torch.cuda.empty_cache()
    return total_ms, realloc_count, peak_gb


def run_paged(layers, cfg, batch, context_len, decode_len, block_size, pool_blocks,
              dtype, device, use_vllm_kernel):
    scale = cfg["head_dim"] ** -0.5

    kv_caches = [PagedKVCache(batch, cfg["num_kv_heads"], cfg["head_dim"],
                               pool_blocks, block_size, dtype, device)
                 for _ in range(len(layers))]
    for kv in kv_caches:
        kv.seq_len = context_len
        kv.seq_lens.fill_(context_len)

    hidden = torch.randn(batch, 1, cfg["hidden"], dtype=dtype, device=device)
    timer = CudaTimer()

    for _ in range(3):
        decode_step_paged(hidden, layers, kv_caches, scale, block_size, use_vllm_kernel)
    torch.cuda.synchronize()
    for kv in kv_caches:
        kv.seq_len = context_len
        kv.seq_lens.fill_(context_len)

    timer.start()
    for step in range(1, decode_len + 1):
        decode_step_paged(hidden, layers, kv_caches, scale, block_size, use_vllm_kernel)
    total_ms = timer.stop()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del kv_caches; torch.cuda.empty_cache()
    return total_ms, 0, peak_gb


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E2E Model: Paged vs BMC on MI300x")
    p.add_argument("--model-config", choices=list(MODEL_CONFIGS.keys()), default="llama-3-8b")
    p.add_argument("--num-layers", type=int, default=0, help="0 = use model default")
    p.add_argument("--context-lengths", type=int, nargs="+", default=[128, 1920])
    p.add_argument("--decode-lengths", type=int, nargs="+", default=[128, 1920])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--pool-blocks", type=int, default=0, help="0 = auto (4x actual need)")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--mode", choices=["paged", "bmc", "both"], default="both")
    p.add_argument("--num-runs", type=int, default=3)
    args = p.parse_args()

    cfg = dict(MODEL_CONFIGS[args.model_config])
    if args.num_layers > 0:
        cfg["num_layers"] = args.num_layers
    dtype = DTYPE_MAP[args.dtype]
    device = "cuda"

    if not torch.cuda.is_available():
        sys.exit("ERROR: No GPU.")

    props = torch.cuda.get_device_properties(0)
    print("=" * 95)
    print("  E2E Model Benchmark: Paged Attention vs BMC (Contiguous SDPA)")
    print("=" * 95)
    print(f"  GPU:          {props.name}")
    print(f"  Model:        {args.model_config} ({cfg['num_layers']}L, "
          f"H={cfg['num_heads']}, KV={cfg['num_kv_heads']}, D={cfg['head_dim']})")
    print(f"  Hidden:       {cfg['hidden']},  FFN inter: {cfg['inter']}")
    print(f"  dtype:        {args.dtype}")
    print(f"  batches:      {args.batch_sizes}")
    print(f"  block_size:   {args.block_size}")
    print(f"  runs:         {args.num_runs}")

    use_vllm_kernel = probe_paged_attention(dtype)
    _probe_gqa(cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"], dtype, device)
    print(f"  GQA mode:     {_GQA_MODE}")
    print(f"  Paged:        {'vLLM paged_attention_v1' if use_vllm_kernel else 'gather+SDPA fallback'}")
    print(f"  BMC:          F.scaled_dot_product_attention (contiguous)")
    print("=" * 95)

    print(f"\n  Building {cfg['num_layers']} transformer layers...")
    layers = [TransformerLayer(cfg["hidden"], cfg["inter"],
                                cfg["num_heads"], cfg["num_kv_heads"],
                                cfg["head_dim"], device, dtype)
              for _ in range(cfg["num_layers"])]
    torch.cuda.synchronize()
    print(f"  Weights mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    modes = ["paged", "bmc"] if args.mode == "both" else [args.mode]

    USECASE_MAP = {
        (128, 128):   "Chat",
        (128, 1920):  "Q&A",
        (1920, 128):  "Summarize",
        (1920, 1920): "Long-form",
    }

    for batch_size in args.batch_sizes:
        print(f"\n  {'═' * 100}")
        print(f"  Batch size = {batch_size}")
        print(f"  {'═' * 100}")
        print(f"  {'UseCase':<12} {'In':>5} {'Out':>5} {'BS':>4} │ "
              f"{'Paged(ms)':>10} {'BMC(ms)':>10} {'Speedup':>8} │ "
              f"{'P tok/s':>9} {'B tok/s':>9} {'#RA':>5} │ "
              f"{'P_GB':>5} {'B_GB':>5}")
        print(f"  {'─' * 100}")

        for ctx in args.context_lengths:
            for dec in args.decode_lengths:
                usecase = USECASE_MAP.get((ctx, dec), "Custom")
                results = {}
                for mode in modes:
                    run_times = []
                    reallocs = 0
                    last_peak = 0
                    for r in range(args.num_runs):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        if mode == "bmc":
                            ms, ra, peak = run_bmc(layers, cfg, batch_size, ctx, dec, dtype, device)
                            reallocs = ra // cfg["num_layers"]
                        else:
                            actual = math.ceil((ctx + dec) / args.block_size) * batch_size
                            pb = args.pool_blocks if args.pool_blocks > 0 else max(actual * 4, 2048)
                            ms, ra, peak = run_paged(layers, cfg, batch_size, ctx, dec,
                                                      args.block_size, pb, dtype, device, use_vllm_kernel)
                        run_times.append(ms)
                        last_peak = peak
                    results[mode] = {
                        "ms": statistics.mean(run_times),
                        "std": statistics.stdev(run_times) if len(run_times) > 1 else 0,
                        "reallocs": reallocs,
                        "peak_gb": last_peak,
                    }

                paged_ms = results.get("paged", {}).get("ms", 0)
                bmc_ms = results.get("bmc", {}).get("ms", 0)
                paged_tps = (dec * batch_size) / (paged_ms / 1000) if paged_ms > 0 else 0
                bmc_tps = (dec * batch_size) / (bmc_ms / 1000) if bmc_ms > 0 else 0
                speedup = paged_ms / bmc_ms if bmc_ms > 0 and paged_ms > 0 else 0
                ra = results.get("bmc", {}).get("reallocs", 0)
                p_gb = results.get("paged", {}).get("peak_gb", 0)
                b_gb = results.get("bmc", {}).get("peak_gb", 0)

                p_str = f"{paged_ms:.1f}" if paged_ms > 0 else "—"
                b_str = f"{bmc_ms:.1f}" if bmc_ms > 0 else "—"
                pt_str = f"{paged_tps:.0f}" if paged_tps > 0 else "—"
                bt_str = f"{bmc_tps:.0f}" if bmc_tps > 0 else "—"
                s_str = f"{speedup:.3f}x" if speedup > 0 else "—"

                print(f"  {usecase:<12} {ctx:>5} {dec:>5} {batch_size:>4} │ "
                      f"{p_str:>10} {b_str:>10} {s_str:>8} │ "
                      f"{pt_str:>9} {bt_str:>9} {ra:>5} │ "
                      f"{p_gb:>4.1f}G {b_gb:>4.1f}G")

        print(f"  {'─' * 100}")

    print("\nDone.")


if __name__ == "__main__":
    main()
