"""SGLang topology auto-tuner entry point."""
from datetime import datetime, timezone
from typing import Dict, Any

from heavy_prefill_bench.optimizer import AutoTuner
from heavy_prefill_bench.reporter import (
    write_sweep_csv,
    write_metadata,
    warn_if_mixed_pricing_metadata,
)


async def run_autotune(config: Dict[str, Any]) -> None:
    """Run the SGLang bench_offline_throughput sweep to find best topology."""

    # Validate required keys
    required = ["model", "workload", "sweep", "hardware"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    wl = config["workload"]
    for key in ["input_len", "output_len"]:
        if key not in wl:
            raise ValueError(f"Missing required workload key: {key}")

    sweep = config["sweep"]
    if "chunked_prefill_sizes" not in sweep:
        raise ValueError("sweep must contain 'chunked_prefill_sizes' list")
    if "num_prompts" in wl and "num_prompts" in sweep:
        raise ValueError(
            "Specify only one of workload.num_prompts or sweep.num_prompts"
        )
    if "num_prompts" not in wl and "num_prompts" not in sweep:
        raise ValueError(
            "Must specify either workload.num_prompts or sweep.num_prompts"
        )

    if "gpu_hourly_cost_usd" not in config["hardware"]:
        raise ValueError("hardware.gpu_hourly_cost_usd is required")
    try:
        gpu_hourly_cost_usd = float(config["hardware"]["gpu_hourly_cost_usd"])
    except (TypeError, ValueError):
        raise ValueError("hardware.gpu_hourly_cost_usd must be numeric")
    if gpu_hourly_cost_usd <= 0:
        raise ValueError("hardware.gpu_hourly_cost_usd must be > 0")
    config["hardware"]["gpu_hourly_cost_usd"] = gpu_hourly_cost_usd

    tuner = AutoTuner(config)
    results = tuner.run()

    if results:
        output_dir = config.get("output_dir", "results")
        csv_path = f"{output_dir}/sglang_autotune.csv"
        write_sweep_csv(csv_path, results, tuner.gpu_hourly_cost_usd)
        print(f"\nResults written to {csv_path}")

        meta = {
            "framework": config.get("framework", "sglang"),
            "model": config["model"],
            "workload": wl,
            "sweep": config["sweep"],
            "sglang_args": config.get("sglang_args", {}),
            "hardware": {
                "gpu_label": tuner.gpu_label,
                "gpu_hourly_cost_usd": tuner.gpu_hourly_cost_usd,
            },
            "pricing": {
                "provider_name": config["hardware"].get("provider_name", "unknown"),
                "instance_type": config["hardware"].get("instance_type", "unknown"),
                "region": config["hardware"].get("region", "unknown"),
                "gpu_hourly_cost_usd": tuner.gpu_hourly_cost_usd,
                "pricing_timestamp_utc": config["hardware"].get(
                    "pricing_timestamp_utc",
                    datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                ),
                "pricing_source_note": config["hardware"].get(
                    "pricing_source_note",
                    "unspecified",
                ),
            },
        }
        write_metadata(f"{output_dir}/sglang_autotune_metadata.json", meta)
        warn_if_mixed_pricing_metadata(output_dir)
