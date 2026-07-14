#!/usr/bin/env python3
"""
FlashInfer PAGED-only block-size sweep: 16 / 32 / 64 / 128 / 256
================================================================
Runs only the paged allocation path (FlashInfer kernel held constant) and sweeps
the page/block size, so you can see how paging overhead vs granularity trades off.

Reuses classes from bench_flashinfer_alloc.py (same model/layers/decode loop).

Usage:
  python bench_paged_blocksize.py --model-config llama-3-8b --batch-sizes 32 128
  python bench_paged_blocksize.py --model-config llama-3-8b --num-layers 4 --block-sizes 16 32 64 128 256
"""
import argparse, statistics, sys
import torch
import bench_flashinfer_alloc as B   # reuse TransformerLayer, KVPaged, run_mode, configs


def main():
    p = argparse.ArgumentParser(description="FlashInfer paged block-size sweep")
    p.add_argument("--model-config", choices=list(B.MODEL_CONFIGS.keys()), default="llama-3-8b")
    p.add_argument("--num-layers", type=int, default=0)
    p.add_argument("--context-lengths", type=int, nargs="+", default=[128, 1920])
    p.add_argument("--decode-lengths", type=int, nargs="+", default=[128, 1920])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128])
    p.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--num-runs", type=int, default=3)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA GPU (FlashInfer requires NVIDIA CUDA).")
    try:
        import flashinfer  # noqa
    except ImportError:
        sys.exit("ERROR: flashinfer not installed. pip install flashinfer-python "
                 "(CUDA-only; will not run on AMD/ROCm).")

    cfg = dict(B.MODEL_CONFIGS[args.model_config])
    if args.num_layers > 0:
        cfg["num_layers"] = args.num_layers
    dtype = B.DTYPE_MAP[args.dtype]; device = "cuda"
    B._FI_WORKSPACE = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    props = torch.cuda.get_device_properties(0)
    blks = args.block_sizes
    print("=" * 96)
    print("  FlashInfer PAGED block-size sweep (kernel held constant)")
    print("=" * 96)
    print(f"  GPU: {props.name} | model: {args.model_config} ({cfg['num_layers']}L, "
          f"H={cfg['num_heads']}, KV={cfg['num_kv_heads']}, D={cfg['head_dim']}) | dtype: {args.dtype}")
    print(f"  block sizes: {blks}")

    layers = [B.TransformerLayer(cfg["hidden"], cfg["inter"], cfg["num_heads"], cfg["num_kv_heads"],
                                 cfg["head_dim"], device, dtype) for _ in range(cfg["num_layers"])]
    torch.cuda.synchronize()

    for bs in args.batch_sizes:
        print(f"\n  {'═'*92}\n  Batch size = {bs}\n  {'═'*92}")
        hdr = f"  {'In':>5} {'Out':>5} {'BS':>4} │ " + " ".join(f"blk{b:>4}(ms)" for b in blks) + \
              " │ " + " ".join(f"{b:>3}GB" for b in blks)
        print(hdr); print(f"  {'─'*(len(hdr))}")
        for ctx in args.context_lengths:
            for dec in args.decode_lengths:
                ms_row, gb_row = [], []
                for blk in blks:
                    times, peak = [], 0.0
                    for _ in range(args.num_runs):
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                        ms, _ra, peak = B.run_mode("paged", layers, cfg, bs, ctx, dec, blk, dtype, device)
                        times.append(ms)
                    ms_row.append(statistics.mean(times)); gb_row.append(peak)
                ms_str = " ".join(f"{m:>9.1f}" for m in ms_row)
                gb_str = " ".join(f"{g:>4.1f}" for g in gb_row)
                print(f"  {ctx:>5} {dec:>5} {bs:>4} │ {ms_str} │ {gb_str}")
        print(f"  {'─'*(len(hdr))}")
    print("\nDone.")


if __name__ == "__main__":
    main()
