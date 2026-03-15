#!/usr/bin/env python3
"""
Allocation Sweep: Attention latency vs number of reallocations (T).
Fixed: Llama-3-8B config, BS=8, Decode=1 (Q=1 autoregressive).

Sweeps T = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, N} for each N.
  T=1      → Upfront (one large alloc, zero reallocs, max wasteful compute)
  T=√N     → BMC optimal (balanced)
  T=N      → Iterative (realloc every step, max copy, min wasteful compute)

Reproduces paper Figure 15 on MI300X for Llama-3-8B.

Usage:
  python bench_alloc_sweep.py
  python bench_alloc_sweep.py --seq-lengths 256 512 1024 2048
  python bench_alloc_sweep.py --batch-size 1
"""

import argparse
import math
import statistics
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F

torch.manual_seed(42)

NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_LAYERS = 32
DTYPE = torch.bfloat16
WARMUP = 3
NUM_RUNS = 3


class ContiguousKVCache:
    def __init__(self, batch, capacity, dtype, device):
        self.batch = batch
        self.capacity = capacity
        self.seq_len = 0
        self.dtype = dtype
        self.device = device
        self.k = torch.zeros(batch, NUM_KV_HEADS, capacity, HEAD_DIM, dtype=dtype, device=device)
        self.v = torch.zeros(batch, NUM_KV_HEADS, capacity, HEAD_DIM, dtype=dtype, device=device)

    def append_kv(self, k_new, v_new):
        self.k[:, :, self.seq_len, :] = k_new
        self.v[:, :, self.seq_len, :] = v_new
        self.seq_len += 1

    def kv_full(self):
        return self.k, self.v

    def grow(self, new_cap):
        new_k = torch.zeros(self.batch, NUM_KV_HEADS, new_cap, HEAD_DIM,
                            dtype=self.dtype, device=self.device)
        new_v = torch.zeros(self.batch, NUM_KV_HEADS, new_cap, HEAD_DIM,
                            dtype=self.dtype, device=self.device)
        new_k[:, :, :self.seq_len, :].copy_(self.k[:, :, :self.seq_len, :])
        new_v[:, :, :self.seq_len, :].copy_(self.v[:, :, :self.seq_len, :])
        del self.k, self.v
        self.k, self.v = new_k, new_v
        self.capacity = new_cap


class CudaTimer:
    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self):
        self.e.record(); torch.cuda.synchronize()
        return self.s.elapsed_time(self.e)


def run_sweep_point(batch, context_len, decode_len, num_allocs, q_tokens, device):
    """Run one (N, T) configuration. Returns total_ms."""
    scale = HEAD_DIM ** -0.5

    if num_allocs <= 1:
        grow_chunk = 0
        initial_cap = context_len + decode_len
    elif num_allocs >= decode_len:
        grow_chunk = 1
        initial_cap = context_len + 1
    else:
        grow_chunk = max(1, decode_len // num_allocs)
        initial_cap = context_len + grow_chunk

    kv_caches = [ContiguousKVCache(batch, initial_cap, DTYPE, device)
                 for _ in range(NUM_LAYERS)]
    for kv in kv_caches:
        kv.seq_len = context_len

    q = torch.randn(batch, NUM_HEADS, q_tokens, HEAD_DIM, dtype=DTYPE, device=device)
    k_new = torch.randn(batch, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    v_new = torch.randn(batch, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    timer = CudaTimer()

    def attn_only(kv_caches):
        for kv in kv_caches:
            k_full, v_full = kv.kv_full()
            F.scaled_dot_product_attention(q, k_full, v_full,
                                           scale=scale, is_causal=False,
                                           enable_gqa=True)

    # Warmup (attention only, no seq_len mutation)
    for _ in range(WARMUP):
        attn_only(kv_caches)
    torch.cuda.synchronize()

    # Measure
    timer.start()
    for step in range(decode_len):
        for kv in kv_caches:
            if grow_chunk > 0 and kv.seq_len + 1 > kv.capacity:
                kv.grow(kv.capacity + grow_chunk)
        for kv in kv_caches:
            kv.append_kv(k_new, v_new)
            k_full, v_full = kv.kv_full()
            F.scaled_dot_product_attention(q, k_full, v_full,
                                           scale=scale, is_causal=False,
                                           enable_gqa=True)
    total_ms = timer.stop()

    del kv_caches
    torch.cuda.empty_cache()
    return total_ms


def main():
    p = argparse.ArgumentParser(description="Allocation sweep: latency vs T")
    p.add_argument("--seq-lengths", type=int, nargs="+", default=[256,512,1024,2048])
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-runs", type=int, default=NUM_RUNS)
    args = p.parse_args()

    device = "cuda"
    if not torch.cuda.is_available():
        sys.exit("No GPU")

    props = torch.cuda.get_device_properties(0)
    print("=" * 90)
    print("  Allocation Sweep: Attention Latency vs T (Llama-3-8B, Decode=1)")
    print("=" * 90)
    print(f"  GPU:       {props.name}")
    print(f"  Config:    H={NUM_HEADS}, KV={NUM_KV_HEADS}, D={HEAD_DIM}, L={NUM_LAYERS}")
    print(f"  BS={args.batch_size}, ctx={args.context}, Q=1")
    print(f"  Runs:      {args.num_runs}")
    print("=" * 90)

    for N in args.seq_lengths:
        sqrt_n = int(math.sqrt(N))
        T_values = sorted(set(
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, N, sqrt_n]
        ))
        T_values = [t for t in T_values if t <= N]

        print(f"\n  {'─' * 75}")
        print(f"  N = {N}   (√N = {sqrt_n})")
        print(f"  {'─' * 75}")
        print(f"  {'T':>6} {'grow_chunk':>12} {'Total(ms)':>12} {'Normalized':>12} {'Note':>10}")
        print(f"  {'─' * 75}")

        results = {}
        for T in T_values:
            times = []
            for _ in range(args.num_runs):
                torch.cuda.empty_cache()
                ms = run_sweep_point(args.batch_size, args.context, N, T, 1, device)
                times.append(ms)
            avg = statistics.mean(times)
            results[T] = avg

        baseline = results.get(1, 1.0)
        for T in T_values:
            avg = results[T]
            gc = N // T if T > 0 else N
            norm = avg / baseline
            note = ""
            if T == 1: note = "Upfront"
            elif T == sqrt_n: note = "√N ★"
            elif T == N: note = "Iterative"
            print(f"  {T:>6} {gc:>12} {avg:>12.1f} {norm:>12.3f} {note:>10}")

    print(f"\n  {'─' * 75}")
    print("Done.")


if __name__ == "__main__":
    main()
