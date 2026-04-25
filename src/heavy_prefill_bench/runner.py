"""Sweep orchestrator."""
import asyncio
from typing import List

from heavy_prefill_bench.adapters.vllm_async_adapter import VLLMAsyncAdapter
from heavy_prefill_bench.adapters.sglang_adapter import SGLangAdapter
from heavy_prefill_bench.generator import generate_dataset
from heavy_prefill_bench.harness import BenchmarkResult, Harness
from heavy_prefill_bench.monitor import GPUMonitor
from heavy_prefill_bench.reporter import write_csv, write_metadata


async def run_sweep(
    framework: str,
    model: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    chunk_sizes: List[int],
    max_seqs_list: List[int],
    concurrencies: List[int],
    output_dir: str = "results",
    max_model_len: int = 52000,
) -> List[BenchmarkResult]:
    """Run a full parameter sweep for a single framework."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    prompts = generate_dataset(num_requests, input_len)
    print(f"Generated {num_requests} prompts of {input_len} tokens each.")

    all_results: List[BenchmarkResult] = []

    for chunk_size in chunk_sizes:
        for max_seqs in max_seqs_list:
            for concurrency in concurrencies:
                print(f"\n=== Sweep: framework={framework}, chunk={chunk_size}, "
                      f"max_seqs={max_seqs}, concurrency={concurrency} ===")

                if framework == "vllm":
                    adapter = VLLMAsyncAdapter(
                        model=model,
                        max_model_len=max_model_len,
                        max_num_batched_tokens=chunk_size,
                        max_num_seqs=max_seqs,
                        trust_remote_code=True,
                    )
                elif framework == "sglang":
                    adapter = SGLangAdapter(
                        model=model,
                        port=8001,
                        chunked_prefill_size=chunk_size,
                        max_running_requests=max_seqs,
                        trust_remote_code=True,
                    )
                else:
                    raise ValueError(f"Unknown framework: {framework}")

                monitor = GPUMonitor(interval_sec=1.0, device_id=0)
                harness = Harness(adapter, concurrency=concurrency, gpu_monitor=monitor)

                await adapter.start({})
                try:
                    result = await harness.run(prompts, output_len=output_len)
                    result.chunk_size = chunk_size
                    result.max_seqs = max_seqs
                    all_results.append(result)

                    successes = [r for r in result.request_results if r.error is None]
                    failures = [r for r in result.request_results if r.error is not None]
                    if successes:
                        avg_ttft = sum(r.ttft_ms for r in successes) / len(successes)
                        avg_tpot = sum(r.tpot_ms for r in successes) / len(successes)
                        avg_e2e = sum(r.e2e_ms for r in successes) / len(successes)
                        print(f"  Successes: {len(successes)}, Failures: {len(failures)}")
                        print(f"  Avg TTFT: {avg_ttft:.1f}ms, Avg TPOT: {avg_tpot:.2f}ms, "
                              f"Avg E2E: {avg_e2e:.1f}ms")
                        print(f"  Wall time: {result.wall_time_sec:.1f}s")
                        if result.gpu_metrics:
                            print(f"  GPU util mean: {result.gpu_metrics.mean_util():.1f}%, "
                                  f"peak VRAM: {result.gpu_metrics.peak_vram_gb():.2f} GB")
                    else:
                        print(f"  All {len(failures)} requests failed.")
                        for f in failures[:3]:
                            print(f"    Error: {f.error}")
                finally:
                    await adapter.stop()

    csv_path = os.path.join(output_dir, f"{framework}.csv")
    write_csv(csv_path, all_results)
    print(f"\nResults written to {csv_path}")

    meta = {
        "framework": framework,
        "model": model,
        "max_model_len": max_model_len,
        "num_requests": num_requests,
        "input_len": input_len,
        "output_len": output_len,
        "chunk_sizes": chunk_sizes,
        "max_seqs_list": max_seqs_list,
        "concurrencies": concurrencies,
    }
    write_metadata(os.path.join(output_dir, f"{framework}_metadata.json"), meta)

    return all_results
