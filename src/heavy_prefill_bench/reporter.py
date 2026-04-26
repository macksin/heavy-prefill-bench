"""CSV reporter for sweep-level benchmark results."""
import csv
import json
import os
from typing import Any, Dict, List


SWEEP_FIELDNAMES = [
    "framework",
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
]


def write_sweep_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write sweep results to CSV (one row per configuration)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SWEEP_FIELDNAMES})


def write_metadata(path: str, metadata: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
