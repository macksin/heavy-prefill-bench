"""Step 7: Medium validation — 10 requests, 5000 input, 200 output."""
import asyncio
import os
import sys

from heavy_prefill_bench.generator import generate_dataset
from heavy_prefill_bench.adapters.vllm_async_adapter import VLLMAsyncAdapter
from heavy_prefill_bench.harness import Harness
from heavy_prefill_bench.monitor import GPUMonitor
from heavy_prefill_bench.reporter import write_csv


async def main():
    model = "microsoft/Phi-4-mini-instruct"
    max_model_len = 52000
    num_requests = 10
    input_len = 5000
    output_len = 200
    concurrency = 2
    chunk_size = 4096
    max_seqs = 8

    prompts = generate_dataset(num_requests, input_len)
    print(f"Generated {num_requests} prompts of {input_len} tokens.")

    adapter = VLLMAsyncAdapter(
        model=model,
        max_model_len=max_model_len,
        max_num_batched_tokens=chunk_size,
        max_num_seqs=max_seqs,
        trust_remote_code=True,
    )
    monitor = GPUMonitor(interval_sec=1.0)
    harness = Harness(adapter, concurrency=concurrency, gpu_monitor=monitor)

    await adapter.start({})
    try:
        result = await harness.run(prompts, output_len=output_len)
        result.chunk_size = chunk_size
        result.max_seqs = max_seqs
        result.framework = "vllm"

        print(f"\n=== Validation Results (Medium) ===")
        print(f"Wall time: {result.wall_time_sec:.2f}s")
        successes = [r for r in result.request_results if r.error is None]
        failures = [r for r in result.request_results if r.error is not None]

        for rr in result.request_results:
            print(f"  req={rr.request_id}: ttft={rr.ttft_ms:.1f}ms, "
                  f"tpot={rr.tpot_ms:.2f}ms, e2e={rr.e2e_ms:.1f}ms, "
                  f"out_toks={rr.output_tokens}, err={rr.error!r}")

        if successes:
            ttfts = sorted(r.ttft_ms for r in successes)
            tpots = sorted(r.tpot_ms for r in successes)
            e2es = sorted(r.e2e_ms for r in successes)
            def p(arr, k): return arr[min(int(len(arr) * k), len(arr) - 1)]
            print(f"\n  TTFT: avg={sum(ttfts)/len(ttfts):.1f}ms, p50={p(ttfts,0.5):.1f}, "
                  f"p90={p(ttfts,0.9):.1f}, p95={p(ttfts,0.95):.1f}")
            print(f"  TPOT: avg={sum(tpots)/len(tpots):.2f}ms, p50={p(tpots,0.5):.1f}, "
                  f"p90={p(tpots,0.9):.1f}, p95={p(tpots,0.95):.1f}")
            print(f"  E2E:  avg={sum(e2es)/len(e2es):.1f}ms, p50={p(e2es,0.5):.1f}, "
                  f"p90={p(e2es,0.9):.1f}, p95={p(e2es,0.95):.1f}")

        if result.gpu_metrics:
            print(f"\n  GPU mean util: {result.gpu_metrics.mean_util():.1f}%")
            print(f"  GPU p95 util: {result.gpu_metrics.p95_util():.1f}%")
            print(f"  Peak VRAM: {result.gpu_metrics.peak_vram_gb():.2f} GB")
            print(f"  GPU samples: {len(result.gpu_metrics.samples)}")

        print(f"\n  Successes: {len(successes)}, Failures: {len(failures)}")

        os.makedirs("results", exist_ok=True)
        write_csv("results/validation_medium.csv", [result])
        print("Wrote results/validation_medium.csv")
    finally:
        await adapter.stop()


if __name__ == "__main__":
    asyncio.run(main())
