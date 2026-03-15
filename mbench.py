
#!/usr/bin/env python3


Compares two fundamentally different KV cache strategies:

  paged  - vLLM default: scattered block allocation + paged attention kernel.
           No reallocation needed, but every attention step pays for block-table
           indirection and non-contiguous memory access.

  bmc    - Block-Managed Contiguous: maintains a contiguous KV tensor so it
           can directly leverage Flash Attention (torch SDPA).  The cache is
           pre-allocated with sqrt(N) headroom; when capacity is exhausted
           (every ~sqrt(N) decode steps), a new contiguous tensor is allocated
           and the live data is copied over.  The bet is that Flash Attention
           speed > periodic realloc+copy cost.

Usage:
  # Default sweep (row conflicts ON, smaller ctx + longer decode)
  python mbench.py --mode both

  # GQA model
  python mbench.py --model-config llama-3-8b

  # Custom sweep
  python mbench.py \\
      --context-lengths 512 1024 2048 4096 8192 \\
      --decode-lengths  1024 2048 4096 \\
      --batch-size 8

  python mbench.py --dtype float16
  python mbench.py --no-row-conflicts
  python mbench.py --csv results.csv
"""

import argparse
import csv
import math
import random
import statistics
import sys
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


HBM3_ROW_SIZE_BYTES = 2048
HBM3_NUM_BANKS_PER_CHANNEL = 16
HBM3_NUM_CHANNELS = 8

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# vLLM's benchmark_paged_attention.py uses 128*1024.
# Configurable via --pool-blocks to test different scatter ranges.
DEFAULT_POOL_BLOCKS = 128 * 1024

MODEL_CONFIGS = {
    "llama-2-7b":  {"num_heads": 32, "num_kv_heads": 32, "head_dim": 128, "num_layers": 32},
    "llama-2-13b": {"num_heads": 40, "num_kv_heads": 40, "head_dim": 128, "num_layers": 40},
    "llama-3-8b":  {"num_heads": 32, "num_kv_heads": 8,  "head_dim": 128, "num_layers": 32},
    "llama-3-70b": {"num_heads": 64, "num_kv_heads": 8,  "head_dim": 128, "num_layers": 80},
    "qwen-7b":     {"num_heads": 32, "num_kv_heads": 32, "head_dim": 128, "num_layers": 32},
    "opt-1.3b":    {"num_heads": 32, "num_kv_heads": 32, "head_dim": 64,  "num_layers": 24},
}


@dataclass
class BenchResult:
    mode: str
    execution_mode: str
    dtype: str
    context_len: int
    decode_len: int
    batch_size: int
    block_size: int
    num_runs: int
    prefill_ms_mean: float = 0.0
    prefill_ms_std: float = 0.0
    decode_total_ms_mean: float = 0.0
    decode_total_ms_std: float = 0.0
    decode_per_token_us_mean: float = 0.0
    decode_per_token_us_std: float = 0.0
    decode_tok_per_s_mean: float = 0.0
    decode_tok_per_s_std: float = 0.0
    realloc_count: int = 0
    realloc_alloc_ms_mean: float = 0.0
    realloc_alloc_ms_std: float = 0.0
    realloc_copy_ms_mean: float = 0.0
    realloc_copy_ms_std: float = 0.0
    decode_compute_ms_mean: float = 0.0
    decode_compute_ms_std: float = 0.0
    peak_mem_gb: float = 0.0
    l2_hits: int = -1
    l2_misses: int = -1
    l2_hit_rate: float = -1.0


# ───────────────────────────────────────────────────────────────────────
#  GPU Utilities
# ───────────────────────────────────────────────────────────────────────

def detect_gpu():
    if not torch.cuda.is_available():
        sys.exit("ERROR: No GPU detected. Run inside a ROCm/CUDA-enabled vLLM docker.")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU:    {props.name}")
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"  VRAM:   {vram / 1e9:.1f} GB")
    print(f"  SMs:    {props.multi_processor_count}")
    if "mi300" not in props.name.lower() and "instinct" not in props.name.lower():
        print(f"  NOTE: Expected MI300X, detected '{props.name}'. Results still valid.")
    return props


def reset_peak_mem():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


class CudaTimer:
    def __init__(self):
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.end_ev = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_ev.record()

    def stop(self) -> float:
        self.end_ev.record()
        torch.cuda.synchronize()
        return self.start_ev.elapsed_time(self.end_ev)


# ───────────────────────────────────────────────────────────────────────
#  Paged KV Cache creation  (matches vLLM's create_kv_caches_with_random)
# ───────────────────────────────────────────────────────────────────────

def create_paged_kv_cache(num_blocks: int, block_size: int,
                          num_kv_heads: int, head_dim: int,
                          dtype: torch.dtype, device: str):
    """
    Create KV cache tensors matching vLLM's exact layout.
    Key:   [num_blocks, num_kv_heads, head_dim // x, block_size, x]
    Value: [num_blocks, num_kv_heads, head_dim, block_size]
    where  x = 16 // element_size   (8 for bf16/fp16)
    """
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    scale = head_dim ** -0.5

    k_cache = torch.empty(
        num_blocks, num_kv_heads, head_dim // x, block_size, x,
        dtype=dtype, device=device).uniform_(-scale, scale)

    v_cache = torch.empty(
        num_blocks, num_kv_heads, head_dim, block_size,
        dtype=dtype, device=device).uniform_(-scale, scale)

    return k_cache, v_cache, x


def create_block_tables(batch_size: int, max_blocks_per_seq: int,
                        num_blocks_pool: int, device: str,
                        seed: int):
    """
    Create block tables matching vLLM's benchmark:
    each slot is random.randint(0, NUM_BLOCKS - 1).
    Guarantees maximum scatter across the full pool address range.
    """
    rng = random.Random(seed)
    tables = [[rng.randint(0, num_blocks_pool - 1)
               for _ in range(max_blocks_per_seq)]
              for _ in range(batch_size)]
    return torch.tensor(tables, dtype=torch.int32, device=device)


# ───────────────────────────────────────────────────────────────────────
#  Contiguous KV Cache  (for BMC)
# ───────────────────────────────────────────────────────────────────────

class ContiguousKVCache:
    """
    Single contiguous tensor per batch, directly usable by Flash Attention.
    Layout: [B, num_kv_heads, capacity, head_dim]  (SDPA-native BHSD)

    grow() reallocates to a larger contiguous buffer and copies live data.
    kv_slices() returns CONTIGUOUS slices (via .contiguous()) so that SDPA
    dispatches to Flash Attention, not the slow math fallback.
    """

    def __init__(self, batch_size: int, num_kv_heads: int, head_dim: int,
                 initial_capacity: int, dtype: torch.dtype, device: str):
        self.batch = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.capacity = initial_capacity
        self.seq_len = 0
        self.dtype = dtype
        self.device = device

        self.k = torch.randn(batch_size, num_kv_heads, initial_capacity, head_dim,
                             dtype=dtype, device=device)
        self.v = torch.randn(batch_size, num_kv_heads, initial_capacity, head_dim,
                             dtype=dtype, device=device)

    def grow(self, new_capacity: int):
        """
        Reallocate to larger buffer, copy live data.
        Returns (alloc_start, alloc_end, copy_start, copy_end) CUDA events.
        Caller queries elapsed AFTER final synchronize — no mid-loop pipeline drain.
        """
        a_s = torch.cuda.Event(enable_timing=True)
        a_e = torch.cuda.Event(enable_timing=True)
        c_s = torch.cuda.Event(enable_timing=True)
        c_e = torch.cuda.Event(enable_timing=True)

        a_s.record()
        new_k = torch.empty(self.batch, self.num_kv_heads, new_capacity, self.head_dim,
                            dtype=self.dtype, device=self.device)
        new_v = torch.empty(self.batch, self.num_kv_heads, new_capacity, self.head_dim,
                            dtype=self.dtype, device=self.device)
        a_e.record()

        c_s.record()
        new_k[:, :, :self.seq_len, :].copy_(self.k[:, :, :self.seq_len, :])
        new_v[:, :, :self.seq_len, :].copy_(self.v[:, :, :self.seq_len, :])
        c_e.record()

        del self.k, self.v
        self.k = new_k
        self.v = new_v
        self.capacity = new_capacity
        return a_s, a_e, c_s, c_e

    def kv_full(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the FULL capacity tensors [B, H, capacity, D] for attention.

        Why not slice to seq_len?  Slicing creates a non-contiguous view,
        and .contiguous() would copy hundreds of MB per decode step — a
        massive artificial overhead that doesn't exist in real implementations
        (real kernels take a length parameter, not a Python slice).

        The extra sqrt(N) headroom tokens are <1.3% of total compute and
        don't affect timing.  After grow(), the tensors are always fresh
        contiguous allocations.
        """
        return self.k, self.v


# ───────────────────────────────────────────────────────────────────────
#  Row-Conflict Estimation
# ───────────────────────────────────────────────────────────────────────

def _run_l2_probe(mode: str, pa_fn, flash_fn, context_len: int, decode_len: int,
                  batch_size: int, block_size: int, num_heads: int,
                  num_kv_heads: int, head_dim: int, pool_blocks: int,
                  dtype: torch.dtype, seed: int, num_iters: int = 50):
    """
    Run a short attention-only loop (no timing, no reporting).
    Designed to be called under rocprof so hardware PMCs are captured.
    """
    device = "cuda"
    scale = 1.0 / math.sqrt(head_dim)

    if mode == "paged":
        max_blk = math.ceil(context_len / block_size)
        actual = max_blk * batch_size
        pb = pool_blocks if pool_blocks > 0 else max(actual * 4, 2048)
        k_c, v_c, _ = create_paged_kv_cache(
            pb, block_size, num_kv_heads, head_dim, dtype, device)
        bt = create_block_tables(batch_size, max_blk, pb, device, seed)
        sl = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)
        q = torch.empty(batch_size, num_heads, head_dim,
                         dtype=dtype, device=device).uniform_(-scale, scale)
        torch.cuda.synchronize()
        for _ in range(num_iters):
            pa_fn(q, k_c, v_c, bt, sl, num_kv_heads, scale, block_size)
        torch.cuda.synchronize()
    else:
        grow_chunk = max(1, int(math.sqrt(context_len + decode_len)))
        cap = context_len + grow_chunk
        kv = ContiguousKVCache(batch_size, num_kv_heads, head_dim,
                               cap, dtype, device)
        kv.seq_len = context_len
        k_f, v_f = kv.kv_full()
        q = torch.empty(batch_size, num_heads, head_dim,
                         dtype=dtype, device=device).uniform_(-scale, scale)
        torch.cuda.synchronize()
        for _ in range(num_iters):
            flash_fn(q, k_f, v_f, num_kv_heads, scale)
        torch.cuda.synchronize()


def run_rocprof_l2(mode: str, context_len: int, decode_len: int,
                   batch_size: int, block_size: int, model_config: str,
                   pool_blocks: int, dtype_str: str, seed: int,
                   paged_kernel: str) -> Dict:
    """
    Invoke rocprof to measure real L2 (TCC) hit/miss hardware counters.
    Runs a short probe of the attention kernel under rocprof, parses CSV.
    """
    import tempfile
    import os
    import subprocess

    rocprof = "/opt/rocm/bin/rocprof"
    if not os.path.isfile(rocprof):
        for p in ["/usr/bin/rocprof", "rocprof"]:
            if os.path.isfile(p) or os.system(f"which {p} > /dev/null 2>&1") == 0:
                rocprof = p
                break

    outdir = tempfile.mkdtemp(prefix="bench_l2_")
    outcsv = os.path.join(outdir, "results.csv")

    probe_script = os.path.join(outdir, "probe.py")
    with open(probe_script, "w") as f:
        f.write(f"""
import sys; sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
import torch
from mbench import (
    MODEL_CONFIGS, DTYPE_MAP, create_paged_kv_cache, create_block_tables,
    ContiguousKVCache, _probe_vllm_pa, _pa_vllm, _pa_gather_sdpa,
    flash_attention_decode, _run_l2_probe,
)
import math

cfg = MODEL_CONFIGS['{model_config}']
dtype = DTYPE_MAP['{dtype_str}']
use_vllm = _probe_vllm_pa(dtype)
pa_fn = _pa_vllm if (use_vllm and '{paged_kernel}' != 'gather-sdpa') else _pa_gather_sdpa

_run_l2_probe(
    mode='{mode}', pa_fn=pa_fn, flash_fn=flash_attention_decode,
    context_len={context_len}, decode_len={decode_len},
    batch_size={batch_size}, block_size={block_size},
    num_heads={cfg['num_heads']}, num_kv_heads={cfg['num_kv_heads']},
    head_dim={cfg['head_dim']}, pool_blocks={pool_blocks},
    dtype=dtype, seed={seed}, num_iters=50,
)
""")

    cmd = [rocprof, "--pmc", "TCC_HIT_sum", "TCC_MISS_sum",
           "-o", outcsv, sys.executable, probe_script]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return {"l2_hits": -1, "l2_misses": -1, "l2_hit_rate": -1}
    except subprocess.TimeoutExpired:
        return {"l2_hits": -1, "l2_misses": -1, "l2_hit_rate": -1}

    hits_total, miss_total = 0, 0
    try:
        with open(outcsv) as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                for k, v in row.items():
                    if "TCC_HIT" in k.upper():
                        try: hits_total += int(v)
                        except ValueError: pass
                    if "TCC_MISS" in k.upper():
                        try: miss_total += int(v)
                        except ValueError: pass
    except Exception:
        return {"l2_hits": -1, "l2_misses": -1, "l2_hit_rate": -1}

    total = hits_total + miss_total
    rate = hits_total / total if total > 0 else 0.0
    return {"l2_hits": hits_total, "l2_misses": miss_total, "l2_hit_rate": rate}


# ───────────────────────────────────────────────────────────────────────
#  Attention Kernels
# ───────────────────────────────────────────────────────────────────────

_VLLM_PA_VERIFIED = None


def _probe_vllm_pa(dtype: torch.dtype):
    """
    Probe vLLM paged_attention using the exact same KV layout as
    vLLM's create_kv_caches_with_random (packed key format).
    """
    global _VLLM_PA_VERIFIED
    if _VLLM_PA_VERIFIED is not None:
        return _VLLM_PA_VERIFIED

    try:
        from vllm._custom_ops import paged_attention_v1

        h, d, blk = 1, 128, 16
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        q = torch.randn(1, h, d, dtype=dtype, device="cuda")
        k = torch.empty(4, h, d // x, blk, x, dtype=dtype, device="cuda").uniform_(-0.1, 0.1)
        v = torch.empty(4, h, d, blk, dtype=dtype, device="cuda").uniform_(-0.1, 0.1)
        bt = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        sl = torch.tensor([2 * blk], dtype=torch.int32, device="cuda")
        out = torch.empty_like(q)
        ks = vs = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        paged_attention_v1(
            out, q, k, v, h, 1.0 / math.sqrt(d),
            bt, sl, blk, 2 * blk, None, "auto", ks, vs)
        _VLLM_PA_VERIFIED = True
        print("  Paged kernel: vLLM paged_attention_v1 (smoke-test passed)")
    except Exception as e:
        _VLLM_PA_VERIFIED = False
        print(f"  Paged kernel: gather+SDPA (vLLM ops unusable: {e})")
    return _VLLM_PA_VERIFIED


# ── Paged attention: vLLM fused kernel ─────────────────────────────────

_KV_SCALE = None

def _pa_vllm(query, k_cache, v_cache, bt, sl, num_kv_heads, scale, block_size):
    from vllm._custom_ops import paged_attention_v1
    global _KV_SCALE
    if _KV_SCALE is None or _KV_SCALE.device != query.device:
        _KV_SCALE = torch.tensor(1.0, dtype=torch.float32, device=query.device)
    out = torch.empty_like(query)
    paged_attention_v1(out, query, k_cache, v_cache,
                       num_kv_heads, scale, bt, sl,
                       block_size, int(sl.max().item()),
                       None, "auto", _KV_SCALE, _KV_SCALE)
    return out


# ── Paged attention: gather blocks → SDPA (always works) ──────────────

def _pa_gather_sdpa(query, k_cache, v_cache, bt, sl, num_kv_heads, scale, block_size):
    """
    Gather scattered KV blocks then run SDPA.
    Handles vLLM's packed key layout: [N, H, D//x, B, x].
    Uses batched tensor indexing (single GPU gather) — no Python loops.
    """
    batch, num_heads, head_dim = query.shape
    gqa = num_heads // num_kv_heads
    max_s = int(sl.max().item())
    max_blk = math.ceil(max_s / block_size)
    idx = bt[:, :max_blk].long()

    # k_cache packed: [N, H_kv, D//x, block_size, x]
    # gather → [batch, max_blk, H_kv, D//x, block_size, x]
    k_g = k_cache[idx]
    # permute → [batch, H_kv, max_blk, block_size, D//x, x]
    k_g = k_g.permute(0, 2, 1, 4, 3, 5).contiguous()
    k_g = k_g.view(batch, num_kv_heads, max_blk * block_size, head_dim)
    k_g = k_g[:, :, :max_s, :]

    # v_cache: [N, H_kv, D, block_size]
    # gather → [batch, max_blk, H_kv, D, block_size]
    v_g = v_cache[idx]
    # permute → [batch, H_kv, max_blk, block_size, D]
    v_g = v_g.permute(0, 2, 1, 4, 3).contiguous()
    v_g = v_g.view(batch, num_kv_heads, max_blk * block_size, head_dim)
    v_g = v_g[:, :, :max_s, :]

    q = query.unsqueeze(2)  # [B, H_q, 1, D]

    if gqa == 1:
        out = F.scaled_dot_product_attention(
            q, k_g, v_g, scale=scale, is_causal=False)
        return out.squeeze(2)

    try:
        out = F.scaled_dot_product_attention(
            q, k_g, v_g, scale=scale, is_causal=False, enable_gqa=True)
        return out.squeeze(2)
    except TypeError:
        pass

    S = k_g.shape[2]
    q = q.view(batch, num_kv_heads, gqa, 1, head_dim)
    k_g = k_g.unsqueeze(2)
    v_g = v_g.unsqueeze(2)
    out = F.scaled_dot_product_attention(
        q, k_g, v_g, scale=scale, is_causal=False)
    return out.view(batch, num_heads, head_dim)


# ── Flash / SDPA attention (for BMC, contiguous KV) ───────────────────

def flash_attention_decode(query, k_cont, v_cont, num_kv_heads, scale):
    """
    query:  [B, num_heads, D]
    k_cont: [B, num_kv_heads, S, D]  — must be contiguous
    v_cont: [B, num_kv_heads, S, D]  — must be contiguous

    GQA handled via enable_gqa (PyTorch 2.5+) or reshape broadcast —
    NO repeat_interleave, which would copy the entire KV cache gqa× per call.
    """
    B = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    gqa = num_heads // num_kv_heads
    q = query.unsqueeze(2)  # [B, H_q, 1, D]

    if gqa == 1:
        out = F.scaled_dot_product_attention(
            q, k_cont, v_cont, scale=scale, is_causal=False)
        return out.squeeze(2)

    try:
        out = F.scaled_dot_product_attention(
            q, k_cont, v_cont, scale=scale, is_causal=False, enable_gqa=True)
        return out.squeeze(2)
    except TypeError:
        pass

    # Fallback: reshape so SDPA broadcasts KV heads across query head groups
    S = k_cont.shape[2]
    q = q.view(B, num_kv_heads, gqa, 1, head_dim)
    k = k_cont.unsqueeze(2)     # [B, H_kv, 1, S, D]
    v = v_cont.unsqueeze(2)     # [B, H_kv, 1, S, D]
    out = F.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=False)
    return out.view(B, num_heads, head_dim)


# ── torch.compile wrapper ─────────────────────────────────────────────

def maybe_compile(fn, execution_mode, tag=""):
    """
    Wraps fn with torch.compile when requested.
    Uses mode="default" (not "reduce-overhead") because seq_len changes
    every decode step — CUDA-graph capture in reduce-overhead would
    re-capture per shape, making it slower than eager.
    """
    if execution_mode == "compile":
        print(f"  Compiling {tag} with torch.compile(mode='default', dynamic=True) ...")
        return torch.compile(fn, mode="default", dynamic=True)
    return fn


# ───────────────────────────────────────────────────────────────────────
#  Single-Run Benchmark: Paged
# ───────────────────────────────────────────────────────────────────────

def _run_paged(pa_fn, context_len, decode_len, batch_size, block_size,
               num_heads, num_kv_heads, head_dim, num_layers,
               warmup, measure_conflicts, seed, dtype, pool_blocks=0):
    """
    Paged attention benchmark:
      - Packed key layout [N, H, D//x, B, x]
      - Block tables use random.randint(0, pool_blocks-1)
      - pool_blocks controls scatter range (0 = auto = 4x actual need)
    """
    device = "cuda"
    scale = 1.0 / math.sqrt(head_dim)
    max_blocks_per_seq = math.ceil((context_len + decode_len) / block_size)
    actual_needed = max_blocks_per_seq * batch_size

    if pool_blocks <= 0:
        pool_blocks = max(actual_needed * 4, 2048)

    reset_peak_mem()

    k_cache, v_cache, x_pack = create_paged_kv_cache(
        pool_blocks, block_size, num_kv_heads, head_dim, dtype, device)

    bt_t = create_block_tables(
        batch_size, max_blocks_per_seq, pool_blocks, device, seed)

    # ── Seq lens ──
    sl_t = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)

    query = torch.empty(batch_size, num_heads, head_dim,
                         dtype=dtype, device=device).uniform_(-scale, scale)
    timer = CudaTimer()

    # ── Warmup ──
    for _ in range(warmup):
        for _ in range(num_layers):
            pa_fn(query, k_cache, v_cache, bt_t, sl_t,
                  num_kv_heads, scale, block_size)
    torch.cuda.synchronize()

    # ── Prefill ──
    timer.start()
    for _ in range(num_layers):
        pa_fn(query, k_cache, v_cache, bt_t, sl_t,
              num_kv_heads, scale, block_size)
    prefill_ms = timer.stop()

    # ── Decode: only sl_t changes each step — no Python block-table updates ──
    timer.start()
    for step in range(1, decode_len + 1):
        sl_t.fill_(context_len + step)
        for _ in range(num_layers):
            pa_fn(query, k_cache, v_cache, bt_t, sl_t,
                  num_kv_heads, scale, block_size)
    decode_ms = timer.stop()

    peak = torch.cuda.max_memory_allocated() / 1e9
    del k_cache, v_cache, bt_t
    torch.cuda.empty_cache()

    return {
        "prefill_ms": prefill_ms,
        "decode_total_ms": decode_ms,
        "decode_per_token_us": decode_ms / max(decode_len, 1) * 1000.0,
        "decode_tok_per_s": (decode_len * batch_size) / (decode_ms / 1000.0)
                            if decode_ms > 0 else 0,
        "realloc_count": 0,
        "realloc_alloc_ms": 0.0,
        "realloc_copy_ms": 0.0,
        "decode_compute_ms": decode_ms,
        "peak_mem_gb": peak,
    }


# ───────────────────────────────────────────────────────────────────────
#  Single-Run Benchmark: BMC  (contiguous + Flash Attention)
# ───────────────────────────────────────────────────────────────────────

def _run_bmc(flash_fn, context_len, decode_len, batch_size, block_size,
             num_heads, num_kv_heads, head_dim, num_layers,
             warmup, measure_conflicts, seed, dtype,
             bmc_strategy="sqrt"):
    """
    BMC benchmark with three allocation strategies:
      sqrt      — realloc every sqrt(N) steps (default, amortized)
      iterative — realloc+copy every single step (worst-case overhead)
      upfront   — pre-allocate full context+decode upfront (zero realloc)
    """
    device = "cuda"
    scale = 1.0 / math.sqrt(head_dim)

    if bmc_strategy == "upfront":
        initial_cap = context_len + decode_len
        grow_chunk = 0
    elif bmc_strategy == "iterative":
        initial_cap = context_len + 1
        grow_chunk = 1
    else:
        grow_chunk = max(1, int(math.sqrt(context_len + decode_len)))
        initial_cap = context_len + grow_chunk

    reset_peak_mem()
    kv = ContiguousKVCache(batch_size, num_kv_heads, head_dim,
                           initial_cap, dtype, device)
    kv.seq_len = context_len

    query = torch.empty(batch_size, num_heads, head_dim,
                         dtype=dtype, device=device).uniform_(-scale, scale)
    timer = CudaTimer()

    # Warmup
    k_f, v_f = kv.kv_full()
    for _ in range(warmup):
        for _ in range(num_layers):
            flash_fn(query, k_f, v_f, num_kv_heads, scale)
    torch.cuda.synchronize()

    # Prefill
    k_f, v_f = kv.kv_full()
    timer.start()
    for _ in range(num_layers):
        flash_fn(query, k_f, v_f, num_kv_heads, scale)
    prefill_ms = timer.stop()

    # Decode
    realloc_count = 0
    alloc_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
    copy_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []

    timer.start()
    for step in range(1, decode_len + 1):
        if grow_chunk > 0 and kv.seq_len + 1 > kv.capacity:
            new_cap = kv.capacity + grow_chunk
            a_s, a_e, c_s, c_e = kv.grow(new_cap)
            alloc_events.append((a_s, a_e))
            copy_events.append((c_s, c_e))
            realloc_count += 1
            k_f, v_f = kv.kv_full()

        kv.seq_len += 1

        for _ in range(num_layers):
            flash_fn(query, k_f, v_f, num_kv_heads, scale)
    decode_ms = timer.stop()

    realloc_alloc_ms = sum(s.elapsed_time(e) for s, e in alloc_events)
    realloc_copy_ms = sum(s.elapsed_time(e) for s, e in copy_events)
    decode_compute_ms = decode_ms - realloc_alloc_ms - realloc_copy_ms

    peak = torch.cuda.max_memory_allocated() / 1e9
    del kv
    torch.cuda.empty_cache()

    return {
        "prefill_ms": prefill_ms,
        "decode_total_ms": decode_ms,
        "decode_per_token_us": decode_ms / max(decode_len, 1) * 1000.0,
        "decode_tok_per_s": (decode_len * batch_size) / (decode_ms / 1000.0)
                            if decode_ms > 0 else 0,
        "realloc_count": realloc_count,
        "realloc_alloc_ms": realloc_alloc_ms,
        "realloc_copy_ms": realloc_copy_ms,
        "decode_compute_ms": decode_compute_ms,
        "peak_mem_gb": peak,
    }


# ───────────────────────────────────────────────────────────────────────
#  Multi-Run Wrapper
# ───────────────────────────────────────────────────────────────────────

def bench_config(mode, execution_mode, dtype_str, run_fn,
                 context_len, decode_len,
                 batch_size, block_size, num_heads, num_kv_heads, head_dim,
                 num_layers, warmup, num_runs, seed, dtype):
    runs = []
    for i in range(num_runs):
        m = run_fn(context_len=context_len, decode_len=decode_len,
                   batch_size=batch_size, block_size=block_size,
                   num_heads=num_heads, num_kv_heads=num_kv_heads,
                   head_dim=head_dim, num_layers=num_layers,
                   warmup=warmup, measure_conflicts=False,
                   seed=seed + i, dtype=dtype)
        runs.append(m)

    def avg(k): return statistics.mean(r[k] for r in runs)
    def sd(k):  return statistics.stdev(r[k] for r in runs) if num_runs > 1 else 0.0

    return BenchResult(
        mode=mode, execution_mode=execution_mode, dtype=dtype_str,
        context_len=context_len, decode_len=decode_len,
        batch_size=batch_size, block_size=block_size, num_runs=num_runs,
        prefill_ms_mean=round(avg("prefill_ms"), 2),
        prefill_ms_std=round(sd("prefill_ms"), 2),
        decode_total_ms_mean=round(avg("decode_total_ms"), 2),
        decode_total_ms_std=round(sd("decode_total_ms"), 2),
        decode_per_token_us_mean=round(avg("decode_per_token_us"), 2),
        decode_per_token_us_std=round(sd("decode_per_token_us"), 2),
        decode_tok_per_s_mean=round(avg("decode_tok_per_s"), 1),
        decode_tok_per_s_std=round(sd("decode_tok_per_s"), 1),
        realloc_count=runs[-1]["realloc_count"],
        realloc_alloc_ms_mean=round(avg("realloc_alloc_ms"), 2),
        realloc_alloc_ms_std=round(sd("realloc_alloc_ms"), 2),
        realloc_copy_ms_mean=round(avg("realloc_copy_ms"), 2),
        realloc_copy_ms_std=round(sd("realloc_copy_ms"), 2),
        decode_compute_ms_mean=round(avg("decode_compute_ms"), 2),
        decode_compute_ms_std=round(sd("decode_compute_ms"), 2),
        peak_mem_gb=round(avg("peak_mem_gb"), 2),
    )


# ───────────────────────────────────────────────────────────────────────
#  Reporting  (paper-minimal)
# ───────────────────────────────────────────────────────────────────────

SEP = "─"


def _fmtl2(n):
    if n < 0:
        return "N/A"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


MODE_LABELS = {
    "paged":       "Paged (PA-v1)",
    "bmc-upfront": "BMC upfront",
    "bmc":         "BMC sqrt(N)",
    "bmc-iter":    "BMC iterative",
}


def print_paper_table(results: Dict[str, BenchResult]):
    """
    Paper-ready table comparing paged against all BMC variants.
    Columns: Attn(ms)  Copy(ms)  Alloc(ms)  Total(ms)  #Reallocs  Speedup
    """
    paged = results.get("paged")
    if not paged or paged.decode_tok_per_s_mean <= 0:
        return

    r0 = paged
    w = 82
    print(f"  {'─' * w}")
    print(f"  ctx={r0.context_len}  dec={r0.decode_len}  BS={r0.batch_size}  "
          f"{r0.dtype}  {r0.num_runs} runs")
    print(f"  {'─' * w}")
    print(f"  {'':>18} {'Attn(ms)':>10} {'Copy(ms)':>10} "
          f"{'Alloc(ms)':>10} {'Total(ms)':>10} {'#RA':>5} {'Speedup':>8}")
    print(f"  {'─' * w}")

    for mode_key in ["paged", "bmc-upfront", "bmc", "bmc-iter"]:
        r = results.get(mode_key)
        if not r:
            continue

        label = MODE_LABELS.get(mode_key, mode_key)
        spd = r.decode_tok_per_s_mean / paged.decode_tok_per_s_mean \
            if paged.decode_tok_per_s_mean > 0 else 0

        if mode_key == "paged":
            print(f"  {label:>18} "
                  f"{r.decode_total_ms_mean:>10.1f} {'—':>10} {'—':>10} "
                  f"{r.decode_total_ms_mean:>10.1f} {'—':>5} {'1.000x':>8}")
        else:
            print(f"  {label:>18} "
                  f"{r.decode_compute_ms_mean:>10.1f} "
                  f"{r.realloc_copy_ms_mean:>10.2f} "
                  f"{r.realloc_alloc_ms_mean:>10.2f} "
                  f"{r.decode_total_ms_mean:>10.1f} "
                  f"{r.realloc_count:>5} "
                  f"{spd:>7.3f}x")

    # L2 cache from rocprof (if available)
    bmc = results.get("bmc")
    if paged.l2_hits >= 0 and bmc and bmc.l2_hits >= 0:
        p_total = paged.l2_hits + paged.l2_misses
        b_total = bmc.l2_hits + bmc.l2_misses
        p_rate = paged.l2_hits / p_total * 100 if p_total > 0 else 0
        b_rate = bmc.l2_hits / b_total * 100 if b_total > 0 else 0
        print(f"  {'─' * w}")
        print(f"  L2 (rocprof)  "
              f"Paged: {_fmtl2(paged.l2_hits)} hits / {_fmtl2(paged.l2_misses)} miss ({p_rate:.1f}%)   "
              f"BMC: {_fmtl2(bmc.l2_hits)} hits / {_fmtl2(bmc.l2_misses)} miss ({b_rate:.1f}%)")

    print(f"  {'─' * w}")


def write_csv(results, path):
    flds = [f.name for f in fields(BenchResult)]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flds)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"\nResults saved to {path}")


# ───────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--context-lengths", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096, 8192],
                   help="default: 512 1024 2048 4096 8192")
    p.add_argument("--decode-lengths", type=int, nargs="+",
                   default=[1024, 2048, 4096],
                   help="default: 1024 2048 4096")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--pool-blocks", type=int, default=0,
                   help="Paged KV pool size in blocks. "
                        "0 = auto (4x actual need, moderate scatter). "
                        "128K = vLLM benchmark default (max scatter). "
                        "Smaller pool → less scatter → faster paged → fairer comparison.")
    p.add_argument("--mode", choices=["paged", "bmc", "bmc-iter", "bmc-upfront", "all", "both"],
                   default="all",
                   help="paged = PA-v1 baseline;  bmc = sqrt(N) realloc;  "
                        "bmc-iter = realloc every step;  bmc-upfront = pre-allocate all;  "
                        "all = paged + all 3 BMC variants;  both = paged + bmc")
    p.add_argument("--dtype", choices=["bfloat16", "float16"],
                   default="bfloat16",
                   help="KV cache and query dtype (default: bfloat16)")
    p.add_argument("--execution-mode", choices=["eager", "compile"],
                   default="eager",
                   help="eager = standard PyTorch;  compile = torch.compile")
    p.add_argument("--paged-kernel", choices=["auto", "vllm", "gather-sdpa"],
                   default="auto",
                   help="auto = vLLM kernel if available, else gather+SDPA; "
                        "vllm = force vLLM paged_attention_v1; "
                        "gather-sdpa = gather blocks then same SDPA as BMC "
                        "(isolates layout effect from kernel quality)")
    p.add_argument("--model-config", choices=list(MODEL_CONFIGS.keys()),
                   default="llama-2-7b")
    p.add_argument("--num-layers", type=int, default=0,
                   help="Override model layer count. 0 = use model default. "
                        "1 = match vLLM's benchmark (fastest).")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--rocprof", action="store_true", default=False,
                   help="Measure real L2 cache hits/misses via rocprof hardware counters")
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    cfg = dict(MODEL_CONFIGS[args.model_config])
    if args.num_layers > 0:
        cfg["num_layers"] = args.num_layers
    dtype = DTYPE_MAP[args.dtype]
    MODE_MAP = {
        "all":   ["paged", "bmc-upfront", "bmc", "bmc-iter"],
        "both":  ["paged", "bmc"],
    }
    modes = MODE_MAP.get(args.mode, [args.mode])

    print("=" * 80)
    print("=" * 80)
    print(f"  Model config:      {args.model_config}")
    print(f"    num_heads:       {cfg['num_heads']},  "
          f"num_kv_heads: {cfg['num_kv_heads']}")
    print(f"    head_dim:        {cfg['head_dim']},  "
          f"num_layers: {cfg['num_layers']}")
    print(f"  dtype:             {args.dtype}")
    print(f"  Block size:        {args.block_size}  (paged mode only)")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Context lengths:   {args.context_lengths}")
    print(f"  Decode lengths:    {args.decode_lengths}")
    print(f"  Modes:             {modes}")
    print(f"  Execution mode:    {args.execution_mode}")
    print(f"  Runs per config:   {args.num_runs}")
    print(f"  Seed:              {args.seed}")
    print(f"  L2 profiling:      {'rocprof' if args.rocprof else 'OFF'}")
    detect_gpu()

    # Resolve paged kernel
    if args.paged_kernel == "vllm":
        if not _probe_vllm_pa(dtype):
            sys.exit("ERROR: --paged-kernel=vllm but vLLM ops failed smoke test")
        pa_base = _pa_vllm
    elif args.paged_kernel == "gather-sdpa":
        _probe_vllm_pa(dtype)
        pa_base = _pa_gather_sdpa
        print("  Paged kernel:      gather+SDPA (same SDPA kernel as BMC)")
    else:
        use_vllm = _probe_vllm_pa(dtype)
        pa_base = _pa_vllm if use_vllm else _pa_gather_sdpa
    fa_base = flash_attention_decode
    pool_label = (f"{args.pool_blocks} blocks" if args.pool_blocks > 0
                  else "auto (4x actual need)")
    print(f"  BMC kernel:        torch.SDPA (Flash Attention)")
    print(f"  Paged kernel mode: {args.paged_kernel}")
    print(f"  Paged pool:        {pool_label}")
    print("=" * 80)

    pa_fn = maybe_compile(pa_base, args.execution_mode, "paged attention")
    fa_fn = maybe_compile(fa_base, args.execution_mode, "flash attention")

    resolved_pool = args.pool_blocks

    def run_paged(**kw):
        return _run_paged(pa_fn, pool_blocks=resolved_pool, **kw)

    def run_bmc(**kw):
        return _run_bmc(fa_fn, bmc_strategy="sqrt", **kw)

    def run_bmc_iter(**kw):
        return _run_bmc(fa_fn, bmc_strategy="iterative", **kw)

    def run_bmc_upfront(**kw):
        return _run_bmc(fa_fn, bmc_strategy="upfront", **kw)

    run_fns = {
        "paged": run_paged,
        "bmc": run_bmc,
        "bmc-iter": run_bmc_iter,
        "bmc-upfront": run_bmc_upfront,
    }

    all_results: List[BenchResult] = []

    for ctx in args.context_lengths:
        for dec in args.decode_lengths:
            config_results: Dict[str, BenchResult] = {}
            for m in modes:
                try:
                    r = bench_config(
                        mode=m, execution_mode=args.execution_mode,
                        dtype_str=args.dtype,
                        run_fn=run_fns[m],
                        context_len=ctx, decode_len=dec,
                        batch_size=args.batch_size, block_size=args.block_size,
                        num_heads=cfg["num_heads"],
                        num_kv_heads=cfg["num_kv_heads"],
                        head_dim=cfg["head_dim"],
                        num_layers=cfg["num_layers"],
                        warmup=args.warmup, num_runs=args.num_runs,
                        seed=args.seed, dtype=dtype)

                    if args.rocprof and m in ("paged", "bmc"):
                        l2 = run_rocprof_l2(
                            mode="paged" if m == "paged" else "bmc",
                            context_len=ctx, decode_len=dec,
                            batch_size=args.batch_size,
                            block_size=args.block_size,
                            model_config=args.model_config,
                            pool_blocks=resolved_pool,
                            dtype_str=args.dtype, seed=args.seed,
                            paged_kernel=args.paged_kernel)
                        r.l2_hits = l2["l2_hits"]
                        r.l2_misses = l2["l2_misses"]
                        r.l2_hit_rate = l2["l2_hit_rate"]

                    all_results.append(r)
                    config_results[m] = r
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f"  ** SKIP {m} ctx={ctx} dec={dec}: {e}")
                    torch.cuda.empty_cache()

            if "paged" in config_results:
                print_paper_table(config_results)
            print()

    # Summary
    pr = [r for r in all_results if r.mode == "paged"]
    if pr:
        print(f"\n  {'═' * 60}")
        print(f"  SUMMARY  ({args.num_runs} runs/config,  {args.dtype})")
        print(f"  {'═' * 60}")
        for bmc_mode in ["bmc-upfront", "bmc", "bmc-iter"]:
            br = [r for r in all_results if r.mode == bmc_mode]
            spds = [b.decode_tok_per_s_mean / p.decode_tok_per_s_mean
                    for p, b in zip(pr, br)
                    if p.decode_tok_per_s_mean > 0 and b.decode_tok_per_s_mean > 0]
            if spds:
                geo = math.exp(sum(math.log(s) for s in spds) / len(spds))
                label = MODE_LABELS.get(bmc_mode, bmc_mode)
                print(f"  {label:<18}  geomean={geo:.3f}x  "
                      f"min={min(spds):.3f}x  max={max(spds):.3f}x  "
                      f"({len(spds)} configs)")
        print(f"  {'═' * 60}")

    if args.csv:
        write_csv(all_results, args.csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
