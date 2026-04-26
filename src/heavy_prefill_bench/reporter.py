"""CSV reporter for sweep-level benchmark results."""
import csv
import json
import os
from typing import Any, Dict, List


SWEEP_FIELDNAMES = [
    "framework",
    "gpu",
    "quantization",
    "chunked_prefill_size",
    "num_prompts",
    "input_len",
    "output_len",
    "requests_per_sec",
    "input_tokens_per_sec",
    "output_tokens_per_sec",
    "total_tokens_per_sec",
    "requests_per_hour",
    "successful_requests",
    "total_output_tokens",
    "model",
    "tp",
    "gpu_hourly_cost_usd",
    "tokens_per_dollar",
]


def write_sweep_csv(path: str, rows: List[Dict[str, Any]], gpu_hourly_cost_usd: float) -> None:
    """Write sweep results to CSV (one row per configuration)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in SWEEP_FIELDNAMES}
            out["gpu_hourly_cost_usd"] = gpu_hourly_cost_usd
            tps = row.get("total_tokens_per_sec", 0)
            if tps and gpu_hourly_cost_usd:
                out["tokens_per_dollar"] = tps * 3600 / gpu_hourly_cost_usd
            else:
                out["tokens_per_dollar"] = ""
            writer.writerow(out)


def write_metadata(path: str, metadata: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
