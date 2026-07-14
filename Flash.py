#!/usr/bin/env python3
"""
FlashInfer allocation-strategy benchmark: Contiguous vs Paged vs BMC
====================================================================

Holds the ATTENTION KERNEL constant (FlashInfer BatchDecodeWithPagedKVCacheWrapper)
and varies ONLY the KV-cache allocation strategy, so any difference is attributable
to allocation/layout, not the kernel:

  contiguous : one big page per sequence, allocated UPFRONT to (context+decode).
               No copy ever; peak memory paid from step 0.
  paged      : fixed-size pages (block_size), grown page-by-page. No copy;
               memory grows per block; non-contiguous page table.
  bmc        : one contiguous page per sequence, grown in ceil(sqrt(N)) chunks
               (realloc + copy once per chunk). Balances copy vs memory.

FlashInfer only attends over the valid seq_len (via last_page_len), so none of the
three waste attention compute on padding — the difference is copy + memory + page-table.

NOTE: written for the current FlashInfer `BatchDecodeWithPagedKVCacheWrapper` API.
      Untested on this machine (no local CUDA). Run on an NVIDIA GPU with flashinfer
      installed. If `plan()` rejects `sm_scale`, the code falls back automatically.

Usage:
  python bench_flashinfer_alloc.py --model-config llama-3-8b
  python bench_flashinfer_alloc.py --model-config llama-3-8b --batch-sizes 1 8 --num-layers 4
"""

import argparse, math, random, statistics, sys
import torch
import torch.nn.functional as F

torch.manual_seed(42); random.seed(42)

MODEL_CONFIGS = {
    "llama-2-7b":  {"num_heads": 32, "num_kv_heads": 32, "head_dim": 128, "num_layers": 32, "hidden": 4096, "inter": 11008},
    "llama-3-8b":  {"num_heads": 32, "num_kv_heads": 8,  "head_dim": 128, "num_layers": 32, "hidden": 4096, "inter": 14336},
    "llama-3-70b": {"num_heads": 64, "num_kv_heads": 8,  "head_dim": 128, "num_layers": 80, "hidden": 8192, "inter": 28672},
    "qwen2-7b":    {"num_heads": 28, "num_kv_heads": 4,  "head_dim": 128, "num_layers": 28, "hidden": 3584, "inter": 18944},
}
DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}

_FI_WORKSPACE = None
_PLAN_TAKES_SCALE = None   # cache whether plan() accepts sm_scale


# ─────────────────────────── transformer layer ───────────────────────────
class RMSNorm:
    def __init__(self, dim, device, dtype):
        self.weight = torch.ones(dim, device=device, dtype=dtype); self.eps = 1e-5
    def __call__(self, x): return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)

class FFN:
    def __init__(self, hidden, inter, device, dtype):
        s = 1.0 / math.sqrt(hidden)
        self.gate = torch.empty(inter, hidden, device=device, dtype=dtype).uniform_(-s, s)
        self.up   = torch.empty(inter, hidden, device=device, dtype=dtype).uniform_(-s, s)
        self.down = torch.empty(hidden, inter, device=device, dtype=dtype).uniform_(-s, s)
    def __call__(self, x): return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)

class QKVProj:
    def __init__(self, hidden, num_heads, num_kv_heads, head_dim, device, dtype):
        s = 1.0 / math.sqrt(hidden)
        self.q_dim = num_heads * head_dim; self.kv_dim = num_kv_heads * head_dim
        self.qkv = torch.empty(self.q_dim + 2 * self.kv_dim, hidden, device=device, dtype=dtype).uniform_(-s, s)
        self.o   = torch.empty(hidden, self.q_dim, device=device, dtype=dtype).uniform_(-s, s)
    def __call__(self, x):
        qkv = F.linear(x, self.qkv)
        return (qkv[..., :self.q_dim], qkv[..., self.q_dim:self.q_dim + self.kv_dim], qkv[..., self.q_dim + self.kv_dim:])

class TransformerLayer:
    def __init__(self, hidden, inter, num_heads, num_kv_heads, head_dim, device, dtype):
        self.norm1 = RMSNorm(hidden, device, dtype); self.norm2 = RMSNorm(hidden, device, dtype)
        self.qkv_proj = QKVProj(hidden, num_heads, num_kv_heads, head_dim, device, dtype)
        self.ffn = FFN(hidden, inter, device, dtype)
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim

class CudaTimer:
    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True); self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self): self.e.record(); torch.cuda.synchronize(); return self.s.elapsed_time(self.e)


# ─────────────────────────── FlashInfer plan helper ───────────────────────────
def _plan(wrapper, indptr, indices, last, H, HKV, D, page_size, dtype, scale):
    """Call plan(), tolerating API variants for the scale kwarg."""
    global _PLAN_TAKES_SCALE
    if _PLAN_TAKES_SCALE is None:
        try:
            wrapper.plan(indptr, indices, last, H, HKV, D, page_size,
                         pos_encoding_mode="NONE", data_type=dtype, q_data_type=dtype, sm_scale=scale)
            _PLAN_TAKES_SCALE = True; return
        except TypeError:
            _PLAN_TAKES_SCALE = False
    if _PLAN_TAKES_SCALE:
        wrapper.plan(indptr, indices, last, H, HKV, D, page_size,
                     pos_encoding_mode="NONE", data_type=dtype, q_data_type=dtype, sm_scale=scale)
    else:
        wrapper.plan(indptr, indices, last, H, HKV, D, page_size,
                     pos_encoding_mode="NONE", data_type=dtype, q_data_type=dtype)


# ─────────────────────────── KV managers (per layer list, shared geometry) ───────────────────────────
class KVBase:
    """Holds one FlashInfer paged-KV tensor per layer + shared page geometry.
    Layout (NHD combined): [num_pages, 2, page_size, num_kv_heads, head_dim]."""
    def __init__(self, num_layers, batch, num_kv_heads, head_dim, context_len, decode_len, dtype, device):
        self.L, self.B = num_layers, batch
        self.HKV, self.D = num_kv_heads, head_dim
        self.dtype, self.device = dtype, device
        self.ctx, self.dec = context_len, decode_len
        self.seq_len = context_len
        self.reallocs = 0

    def maybe_grow(self):  # override in BMC
        pass

    def write(self, i, pos, k, v):  # k,v: [B, HKV, D]
        raise NotImplementedError

    def geometry(self, new_len):    # -> (indptr, indices, last_page_len, page_size)
        raise NotImplementedError

    def kv(self, i):
        return self.kv_list[i]


class KVContiguous(KVBase):
    """One page per sequence, page_size = full capacity, allocated upfront."""
    def __init__(self, *a):
        super().__init__(*a)
        self.cap = self.ctx + self.dec
        self.page_size = self.cap
        self.kv_list = [torch.zeros(self.B, 2, self.cap, self.HKV, self.D, dtype=self.dtype, device=self.device)
                        for _ in range(self.L)]
        self.brange = torch.arange(self.B, device=self.device)
        # geometry is constant across steps except last_page_len -> preallocate once
        self._indptr = torch.arange(self.B + 1, device=self.device, dtype=torch.int32)   # 1 page/seq
        self._indices = torch.arange(self.B, device=self.device, dtype=torch.int32)
        self._last = torch.empty(self.B, dtype=torch.int32, device=self.device)

    def write(self, i, pos, k, v):
        self.kv_list[i][self.brange, 0, pos, :, :] = k
        self.kv_list[i][self.brange, 1, pos, :, :] = v

    def geometry(self, new_len):
        self._last.fill_(new_len)
        return self._indptr, self._indices, self._last, self.page_size


class KVPaged(KVBase):
    """Fixed-size pages (block_size), page-by-page growth, non-contiguous."""
    def __init__(self, *a, block_size=16):
        super().__init__(*a)
        self.page_size = block_size
        self.ppr = math.ceil((self.ctx + self.dec) / block_size)
        self.num_pages = self.B * self.ppr
        self.kv_list = [torch.zeros(self.num_pages, 2, block_size, self.HKV, self.D, dtype=self.dtype, device=self.device)
                        for _ in range(self.L)]
        self.base = torch.arange(self.B, device=self.device, dtype=torch.int32) * self.ppr
        # preallocate geometry buffers once; slice/update in place per step (no per-step alloc)
        ar = torch.arange(self.ppr, device=self.device, dtype=torch.int32)
        self._full_idx = self.base.view(self.B, 1) + ar.view(1, self.ppr)          # [B, ppr] global page ids
        self._indptr_unit = torch.arange(self.B + 1, device=self.device, dtype=torch.int32)
        self._last = torch.empty(self.B, dtype=torch.int32, device=self.device)

    def write(self, i, pos, k, v):
        gp = (self.base + pos // self.page_size).long()
        off = pos % self.page_size
        self.kv_list[i][gp, 0, off, :, :] = k
        self.kv_list[i][gp, 1, off, :, :] = v

    def geometry(self, new_len):
        active = math.ceil(new_len / self.page_size)
        last = ((new_len - 1) % self.page_size) + 1
        indptr = self._indptr_unit * active                        # [B+1]
        indices = self._full_idx[:, :active].reshape(-1)           # active pages/seq, no Python loop
        self._last.fill_(last)
        return indptr, indices, self._last, self.page_size


class KVBmc(KVBase):
    """One contiguous page per sequence; grow capacity by ceil(sqrt(N)) with realloc+copy."""
    def __init__(self, *a):
        super().__init__(*a)
        self.chunk = max(1, int(math.sqrt(self.ctx + self.dec)))
        self.cap = self.ctx + self.chunk
        self.page_size = self.cap
        self.kv_list = [torch.zeros(self.B, 2, self.cap, self.HKV, self.D, dtype=self.dtype, device=self.device)
                        for _ in range(self.L)]
        self.brange = torch.arange(self.B, device=self.device)
        self._indptr = torch.arange(self.B + 1, device=self.device, dtype=torch.int32)
        self._indices = torch.arange(self.B, device=self.device, dtype=torch.int32)
        self._last = torch.empty(self.B, dtype=torch.int32, device=self.device)

    def maybe_grow(self):
        if self.seq_len + 1 > self.cap:
            new_cap = self.cap + self.chunk
            new_list = []
            for kvt in self.kv_list:
                nt = torch.zeros(self.B, 2, new_cap, self.HKV, self.D, dtype=self.dtype, device=self.device)
                nt[:, :, :self.seq_len, :, :].copy_(kvt[:, :, :self.seq_len, :, :])   # BMC copy
                new_list.append(nt)
            del self.kv_list; self.kv_list = new_list
            self.cap = new_cap; self.page_size = new_cap
            self.reallocs += 1

    def write(self, i, pos, k, v):
        self.kv_list[i][self.brange, 0, pos, :, :] = k
        self.kv_list[i][self.brange, 1, pos, :, :] = v

    def geometry(self, new_len):
        self._last.fill_(new_len)
        return self._indptr, self._indices, self._last, self.page_size


# ─────────────────────────── decode step (kernel held constant) ───────────────────────────
def decode_step(hidden, layers, kv, wrapper, scale):
    B = hidden.shape[0]
    kv.maybe_grow()
    pos = kv.seq_len; new_len = pos + 1
    indptr, indices, last, page_size = kv.geometry(new_len)
    L0 = layers[0]
    _plan(wrapper, indptr, indices, last, L0.num_heads, L0.num_kv_heads, L0.head_dim, page_size, kv.dtype, scale)
    for i, layer in enumerate(layers):
        residual = hidden
        x = layer.norm1(hidden)
        q, k_new, v_new = layer.qkv_proj(x)
        kv.write(i, pos, k_new.view(B, layer.num_kv_heads, layer.head_dim),
                        v_new.view(B, layer.num_kv_heads, layer.head_dim))
        o = wrapper.run(q.view(B, layer.num_heads, layer.head_dim), kv.kv(i))   # [B,H,D]
        hidden = F.linear(o.view(B, 1, -1), layer.qkv_proj.o) + residual
        residual = hidden
        hidden = layer.ffn(layer.norm2(hidden)) + residual
    kv.seq_len = new_len
    return hidden


def run_mode(mode, layers, cfg, batch, ctx, dec, block_size, dtype, device):
    import flashinfer
    scale = cfg["head_dim"] ** -0.5
    L = len(layers)
    if mode == "contiguous":
        kv = KVContiguous(L, batch, cfg["num_kv_heads"], cfg["head_dim"], ctx, dec, dtype, device)
    elif mode == "paged":
        kv = KVPaged(L, batch, cfg["num_kv_heads"], cfg["head_dim"], ctx, dec, dtype, device, block_size=block_size)
    else:
        kv = KVBmc(L, batch, cfg["num_kv_heads"], cfg["head_dim"], ctx, dec, dtype, device)

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(_FI_WORKSPACE, "NHD")
    hidden = torch.randn(batch, 1, cfg["hidden"], dtype=dtype, device=device)

    for _ in range(5):
        decode_step(hidden, layers, kv, wrapper, scale)
    torch.cuda.synchronize()
    kv.seq_len = ctx; kv.reallocs = 0

    timer = CudaTimer(); timer.start()
    for _ in range(dec):
        decode_step(hidden, layers, kv, wrapper, scale)
    ms = timer.stop()
    peak = torch.cuda.max_memory_allocated() / 1e9
    reallocs = kv.reallocs
    del kv; torch.cuda.empty_cache()
    return ms, reallocs, peak


# ─────────────────────────── steady-state (plan once, fixed length) ───────────────────────────
def run_steady(mode, layers, cfg, batch, length, block_size, dtype, device, steps=64, warmup=16):
    """Isolate the per-step DECODE cost at a FIXED sequence length.

    Prefills capacity to `length`, plans the FlashInfer wrapper ONCE for that
    geometry, then times `steps` decode iterations that write at the tail slot
    (overwrite) instead of growing. No per-step planning, no allocation, no
    realloc/copy — so the number reflects steady-state kernel + layer cost only.
    This is the clean way to compare paged block sizes without host-side noise.
    """
    import flashinfer
    scale = cfg["head_dim"] ** -0.5
    L = len(layers)
    if mode == "contiguous":
        kv = KVContiguous(L, batch, cfg["num_kv_heads"], cfg["head_dim"], length, 0, dtype, device)
    elif mode == "paged":
        kv = KVPaged(L, batch, cfg["num_kv_heads"], cfg["head_dim"], length, 0, dtype, device, block_size=block_size)
    else:
        kv = KVBmc(L, batch, cfg["num_kv_heads"], cfg["head_dim"], length, 0, dtype, device)
    kv.seq_len = length

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(_FI_WORKSPACE, "NHD")
    indptr, indices, last, page_size = kv.geometry(length)   # plan ONCE for fixed length
    L0 = layers[0]
    _plan(wrapper, indptr, indices, last, L0.num_heads, L0.num_kv_heads, L0.head_dim, page_size, dtype, scale)

    hidden = torch.randn(batch, 1, cfg["hidden"], dtype=dtype, device=device)
    pos = length - 1   # overwrite tail every step -> length stays fixed, plan stays valid

    def one_step(h):
        for i, layer in enumerate(layers):
            residual = h
            x = layer.norm1(h)
            q, k_new, v_new = layer.qkv_proj(x)
            kv.write(i, pos, k_new.view(batch, layer.num_kv_heads, layer.head_dim),
                            v_new.view(batch, layer.num_kv_heads, layer.head_dim))
            o = wrapper.run(q.view(batch, layer.num_heads, layer.head_dim), kv.kv(i))
            h = F.linear(o.view(batch, 1, -1), layer.qkv_proj.o) + residual
            residual = h
            h = layer.ffn(layer.norm2(h)) + residual
        return h

    for _ in range(warmup):
        hidden = one_step(hidden)
    torch.cuda.synchronize()

    timer = CudaTimer(); timer.start()
    for _ in range(steps):
        hidden = one_step(hidden)
    ms = timer.stop()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del kv; torch.cuda.empty_cache()
    return ms / steps, peak   # per-step ms


# ─────────────────────────── main ───────────────────────────
def main():
    global _FI_WORKSPACE
    p = argparse.ArgumentParser(description="FlashInfer: contiguous vs paged vs BMC allocation")
    p.add_argument("--model-config", choices=list(MODEL_CONFIGS.keys()), default="llama-3-8b")
    p.add_argument("--num-layers", type=int, default=0)
    p.add_argument("--context-lengths", type=int, nargs="+", default=[128, 1920])
    p.add_argument("--decode-lengths", type=int, nargs="+", default=[128, 1920])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--steady-lengths", type=int, nargs="+", default=[],
                   help="If set, run FIXED-LENGTH steady-state mode at these lengths "
                        "(plan once, no growth) instead of the growth benchmark.")
    p.add_argument("--steady-steps", type=int, default=64,
                   help="Timed decode steps per config in steady-state mode.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA GPU (FlashInfer requires NVIDIA CUDA).")
    try:
        import flashinfer  # noqa
    except ImportError:
        sys.exit("ERROR: flashinfer not installed. Install with:\n"
                 "  pip install flashinfer-python\n"
                 "  (or a CUDA-matched wheel: pip install flashinfer-python "
                 "-i https://flashinfer.ai/whl/cu124/torch2.6/)\n"
                 "  NOTE: FlashInfer is CUDA-only — it will not run on AMD/ROCm (e.g. MI300x).")

    cfg = dict(MODEL_CONFIGS[args.model_config])
    if args.num_layers > 0: cfg["num_layers"] = args.num_layers
    dtype = DTYPE_MAP[args.dtype]; device = "cuda"
    _FI_WORKSPACE = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    props = torch.cuda.get_device_properties(0)
    print("=" * 92)
    print("  FlashInfer allocation benchmark — Contiguous vs Paged vs BMC (kernel held constant)")
    print("=" * 92)
    print(f"  GPU: {props.name} | model: {args.model_config} ({cfg['num_layers']}L, "
          f"H={cfg['num_heads']}, KV={cfg['num_kv_heads']}, D={cfg['head_dim']}) | dtype: {args.dtype}")

    layers = [TransformerLayer(cfg["hidden"], cfg["inter"], cfg["num_heads"], cfg["num_kv_heads"],
                               cfg["head_dim"], device, dtype) for _ in range(cfg["num_layers"])]
    torch.cuda.synchronize()
    modes = ["contiguous", "paged", "bmc"]

    # ── steady-state mode: plan once, fixed length, per-step latency ──
    if args.steady_lengths:
        print(f"  mode: STEADY-STATE (fixed length, plan once, {args.steady_steps} timed steps/config, "
              f"block_size={args.block_size})")
        for bs in args.batch_sizes:
            print(f"\n  {'═'*70}\n  Batch size = {bs}\n  {'═'*70}")
            print(f"  {'Len':>6} {'BS':>4} │ {'Contig(ms)':>11} {'Paged(ms)':>10} {'BMC(ms)':>9} │ "
                  f"{'C_GB':>5} {'P_GB':>5} {'B_GB':>5}")
            print(f"  {'─'*70}")
            for length in args.steady_lengths:
                res = {}
                for mode in modes:
                    times = []; peak = 0
                    for _ in range(args.num_runs):
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                        ms, peak = run_steady(mode, layers, cfg, bs, length,
                                              args.block_size, dtype, device, steps=args.steady_steps)
                        times.append(ms)
                    res[mode] = (statistics.median(times), peak)
                c_ms, c_gb = res["contiguous"]; p_ms, p_gb = res["paged"]; b_ms, b_gb = res["bmc"]
                print(f"  {length:>6} {bs:>4} │ {c_ms:>11.3f} {p_ms:>10.3f} {b_ms:>9.3f} │ "
                      f"{c_gb:>4.1f}G {p_gb:>4.1f}G {b_gb:>4.1f}G")
            print(f"  {'─'*70}")
        print("\nDone.")
        return

    for bs in args.batch_sizes:
        print(f"\n  {'═'*88}\n  Batch size = {bs}\n  {'═'*88}")
        print(f"  {'In':>5} {'Out':>5} {'BS':>4} │ {'Contig(ms)':>11} {'Paged(ms)':>10} {'BMC(ms)':>9} │ "
              f"{'C_GB':>5} {'P_GB':>5} {'B_GB':>5} {'#RA':>4}")
        print(f"  {'─'*88}")
        for ctx in args.context_lengths:
            for dec in args.decode_lengths:
                res = {}
                for mode in modes:
                    times = []; peak = 0; ra = 0
                    for _ in range(args.num_runs):
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                        ms, ra, peak = run_mode(mode, layers, cfg, bs, ctx, dec,
                                                args.block_size, dtype, device)
                        times.append(ms)
                    res[mode] = (statistics.median(times), peak, ra)
                c_ms, c_gb, _ = res["contiguous"]; p_ms, p_gb, _ = res["paged"]; b_ms, b_gb, b_ra = res["bmc"]
                print(f"  {ctx:>5} {dec:>5} {bs:>4} │ {c_ms:>11.1f} {p_ms:>10.1f} {b_ms:>9.1f} │ "
                      f"{c_gb:>4.1f}G {p_gb:>4.1f}G {b_gb:>4.1f}G {b_ra:>4}")
        print(f"  {'─'*88}")
    print("\nDone.")


if __name__ == "__main__":
    main()
