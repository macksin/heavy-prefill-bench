"""SGLang topology auto-tuner entry point."""
from typing import Dict, Any

from heavy_prefill_bench.optimizer import AutoTuner
from heavy_prefill_bench.reporter import write_sweep_csv, write_metadata


async def run_autotune(config: Dict[str, Any]) -> None:
    """Run the SGLang bench_offline_throughput sweep to find best topology."""

    # Validate required keys
    required = ["model", "workload", "sweep", "hardware"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    wl = config["workload"]
    for key in ["input_len", "output_len", "num_prompts"]:
        if key not in wl:
            raise ValueError(f"Missing required workload key: {key}")

    if "chunked_prefill_sizes" not in config["sweep"]:
        raise ValueError("sweep must contain 'chunked_prefill_sizes' list")

    if "gpu_hourly_cost_usd" not in config["hardware"]:
        raise ValueError("hardware.gpu_hourly_cost_usd is required")

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
        }
        write_metadata(f"{output_dir}/sglang_autotune_metadata.json", meta)
