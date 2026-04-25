"""CSV reporter for benchmark results."""
import csv
import os
from typing import Any, Dict, List

from heavy_prefill_bench.harness import BenchmarkResult


def write_csv(path: str, results: List[BenchmarkResult], append: bool = False) -> None:
    """Write benchmark results to a CSV file with the fixed schema."""
    fieldnames = [
        "request_id",
        "input_tokens",
        "output_tokens",
        "ttft_ms",
        "tpot_ms",
        "e2e_ms",
        "gpu_util_mean",
        "vram_peak_gb",
        "framework",
        "chunk_size",
        "max_seqs",
        "concurrency",
    ]

    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for br in results:
            gpu_mean = br.gpu_metrics.mean_util() if br.gpu_metrics else 0.0
            vram_peak = br.gpu_metrics.peak_vram_gb() if br.gpu_metrics else 0.0
            for rr in br.request_results:
                writer.writerow({
                    "request_id": rr.request_id,
                    "input_tokens": rr.input_tokens,
                    "output_tokens": rr.output_tokens,
                    "ttft_ms": rr.ttft_ms,
                    "tpot_ms": rr.tpot_ms,
                    "e2e_ms": rr.e2e_ms,
                    "gpu_util_mean": round(gpu_mean, 2),
                    "vram_peak_gb": round(vram_peak, 2),
                    "framework": br.framework,
                    "chunk_size": br.chunk_size if br.chunk_size is not None else "",
                    "max_seqs": br.max_seqs if br.max_seqs is not None else "",
                    "concurrency": br.concurrency,
                })


def write_metadata(path: str, metadata: Dict[str, Any]) -> None:
    import json
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
