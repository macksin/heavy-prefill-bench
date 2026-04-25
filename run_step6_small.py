"""Step 6: Small validation — 5 requests, 500 input, 10 output."""
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
    num_requests = 5
    input_len = 500
    output_len = 10
    concurrency = 1
    chunk_size = 4096
    max_seqs = 4

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

        print(f"\n=== Validation Results ===")
        print(f"Wall time: {result.wall_time_sec:.2f}s")
        for rr in result.request_results:
            print(f"  req={rr.request_id}: ttft={rr.ttft_ms:.1f}ms, "
                  f"tpot={rr.tpot_ms:.2f}ms, e2e={rr.e2e_ms:.1f}ms, "
                  f"out_toks={rr.output_tokens}, err={rr.error!r}")
        if result.gpu_metrics:
            print(f"  GPU mean util: {result.gpu_metrics.mean_util():.1f}%")
            print(f"  GPU p95 util: {result.gpu_metrics.p95_util():.1f}%")
            print(f"  Peak VRAM: {result.gpu_metrics.peak_vram_gb():.2f} GB")

        os.makedirs("results", exist_ok=True)
        write_csv("results/validation_small.csv", [result])
        print("\nWrote results/validation_small.csv")
    finally:
        await adapter.stop()


if __name__ == "__main__":
    asyncio.run(main())
