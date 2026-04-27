#!/usr/bin/env python3
"""Generate CSV and metadata from existing JSONL results for the incomplete 7B run."""
import csv
import json
import os
from datetime import datetime, timezone

RESULTS_DIR = "results"
OUTPUT_CSV = f"{RESULTS_DIR}/sglang_autotune.csv"
OUTPUT_META = f"{RESULTS_DIR}/sglang_autotune_metadata.json"

GPU_LABEL = "NVIDIA H100 80GB HBM3"
GPU_COST = 2.99
MODEL = "Qwen/Qwen2.5-7B-Instruct"
FRAMEWORK = "sglang"
TP = 1
QUANTIZATION = "bf16"
INPUT_LEN = 4000
OUTPUT_LEN = 1000

SWEEP_FIELDNAMES = [
    "framework", "gpu", "quantization", "chunked_prefill_size", "num_prompts",
    "input_len", "output_len", "requests_per_sec", "input_tokens_per_sec",
    "output_tokens_per_sec", "total_tokens_per_sec", "requests_per_hour",
    "successful_requests", "total_output_tokens", "model", "tp",
    "gpu_hourly_cost_usd", "tokens_per_dollar", "oom",
]

NUM_PROMPTS_VALUES = [50, 100, 200, 400]
CHUNK_SIZES = [2048, 4096, 8192, 16384, 32768]

rows = []
for num_prompts in NUM_PROMPTS_VALUES:
    for chunk_size in CHUNK_SIZES:
        jsonl_path = f"{RESULTS_DIR}/sglang_p{num_prompts}_chunk{chunk_size}.jsonl"
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                last = None
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            if "request_throughput" in rec:
                                last = rec
                        except json.JSONDecodeError:
                            pass
            if last:
                tps = last.get("total_throughput", 0)
                row = {
                    "framework": FRAMEWORK,
                    "gpu": GPU_LABEL,
                    "quantization": QUANTIZATION,
                    "chunked_prefill_size": chunk_size,
                    "num_prompts": num_prompts,
                    "input_len": INPUT_LEN,
                    "output_len": OUTPUT_LEN,
                    "requests_per_sec": last.get("request_throughput", 0),
                    "input_tokens_per_sec": last.get("input_throughput", 0),
                    "output_tokens_per_sec": last.get("output_throughput", 0),
                    "total_tokens_per_sec": tps,
                    "requests_per_hour": last.get("request_throughput", 0) * 3600,
                    "successful_requests": last.get("successful_requests", 0),
                    "total_output_tokens": last.get("total_output_tokens", 0),
                    "model": MODEL,
                    "tp": TP,
                    "gpu_hourly_cost_usd": GPU_COST,
                    "tokens_per_dollar": tps * 3600 / GPU_COST if tps else "",
                    "oom": False,
                }
                rows.append(row)
                continue

        # Missing or unparsable = OOM/missing
        rows.append({
            "framework": FRAMEWORK,
            "gpu": GPU_LABEL,
            "quantization": QUANTIZATION,
            "chunked_prefill_size": chunk_size,
            "num_prompts": num_prompts,
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "requests_per_sec": "",
            "input_tokens_per_sec": "",
            "output_tokens_per_sec": "",
            "total_tokens_per_sec": "",
            "requests_per_hour": "",
            "successful_requests": "",
            "total_output_tokens": "",
            "model": MODEL,
            "tp": TP,
            "gpu_hourly_cost_usd": GPU_COST,
            "tokens_per_dollar": "",
            "oom": True,
        })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Wrote {OUTPUT_CSV} ({len(rows)} rows)")

# Find best config
successful = [r for r in rows if not r.get("oom")]
if successful:
    best = max(successful, key=lambda r: r["requests_per_sec"])
    print(f"\nBEST CONFIG (from available results):")
    print(f"  num_prompts          = {best['num_prompts']}")
    print(f"  chunked_prefill_size = {best['chunked_prefill_size']}")
    print(f"  requests/hour        = {best['requests_per_hour']:.0f}")
    print(f"  tokens/sec           = {best['total_tokens_per_sec']:.0f}")

# Write metadata
meta = {
    "framework": FRAMEWORK,
    "model": MODEL,
    "workload": {"input_len": INPUT_LEN, "output_len": OUTPUT_LEN},
    "sweep": {"num_prompts": NUM_PROMPTS_VALUES, "chunked_prefill_sizes": CHUNK_SIZES},
    "sglang_args": {"tp": TP, "mem_fraction_static": 0.85, "disable_radix_cache": True, "random_range_ratio": 0.0, "quantization": None},
    "hardware": {
        "gpu_label": GPU_LABEL,
        "gpu_hourly_cost_usd": GPU_COST,
    },
    "pricing": {
        "provider_name": "runpod",
        "instance_type": "1x H100 80GB",
        "region": "us-ca",
        "gpu_hourly_cost_usd": GPU_COST,
        "pricing_timestamp_utc": "2026-04-27T00:00:00+00:00",
        "pricing_source_note": "RunPod on-demand listed price",
    },
    "session_note": "INCOMPLETE RUN. Missing p400_chunk32768 (timeout). 14B and 32B not started.",
    "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}

with open(OUTPUT_META, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Wrote {OUTPUT_META}")
