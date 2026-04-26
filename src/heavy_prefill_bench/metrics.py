"""Throughput-focused metrics for topology optimization."""

from heavy_prefill_bench.harness import BenchmarkResult


def prompts_per_hour(result: BenchmarkResult) -> float:
    """Compute prompts/hour from a benchmark result."""
    if result.wall_time_sec <= 0:
        return 0.0
    successes = [r for r in result.request_results if r.error is None]
    return (len(successes) / result.wall_time_sec) * 3600


def tokens_per_second(result: BenchmarkResult) -> float:
    """Compute total tokens/second (input + output) from a benchmark result."""
    if result.wall_time_sec <= 0:
        return 0.0
    total_input = sum(r.input_tokens for r in result.request_results if r.error is None)
    total_output = sum(r.output_tokens for r in result.request_results if r.error is None)
    return (total_input + total_output) / result.wall_time_sec


def throughput_summary(result: BenchmarkResult) -> dict:
    """Return a dict with all throughput-relevant metrics."""
    successes = [r for r in result.request_results if r.error is None]
    failures = [r for r in result.request_results if r.error is not None]

    summary = {
        "prompts_per_hour": prompts_per_hour(result),
        "tokens_per_second": tokens_per_second(result),
        "wall_time_sec": result.wall_time_sec,
        "num_success": len(successes),
        "num_failure": len(failures),
        "concurrency": result.concurrency,
        "chunk_size": result.chunk_size,
        "max_seqs": result.max_seqs,
    }

    if successes:
        e2es = sorted(r.e2e_ms for r in successes)
        ttfts = sorted(r.ttft_ms for r in successes)
        tpots = sorted(r.tpot_ms for r in successes)

        def p(arr, k):
            return arr[min(int(len(arr) * k), len(arr) - 1)]

        summary["ttft_p50_ms"] = p(ttfts, 0.5)
        summary["ttft_p90_ms"] = p(ttfts, 0.9)
        summary["ttft_p99_ms"] = p(ttfts, 0.99)
        summary["ttft_avg_ms"] = sum(ttfts) / len(ttfts)
        summary["tpot_avg_ms"] = sum(tpots) / len(tpots)
        summary["e2e_p50_ms"] = p(e2es, 0.5)
        summary["e2e_p99_ms"] = p(e2es, 0.99)
        summary["e2e_avg_ms"] = sum(e2es) / len(e2es)

    if result.gpu_metrics:
        summary["gpu_util_mean"] = result.gpu_metrics.mean_util()
        summary["vram_peak_gb"] = result.gpu_metrics.peak_vram_gb()

    return summary
