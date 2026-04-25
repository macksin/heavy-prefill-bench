import asyncio
import sys

import yaml

from heavy_prefill_bench.runner import run_sweep


async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for framework in config["frameworks"]:
        await run_sweep(
            framework=framework,
            model=config["model"],
            num_requests=config["num_requests"],
            input_len=config["input_len"],
            output_len=config["output_len"],
            chunk_sizes=config["chunk_sizes"],
            max_seqs_list=config["max_seqs"],
            concurrencies=config["concurrencies"],
            output_dir=config.get("output_dir", "results"),
            max_model_len=config.get("max_model_len", 52000),
        )


if __name__ == "__main__":
    asyncio.run(main())
