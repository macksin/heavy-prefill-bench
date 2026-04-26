import asyncio
import sys

import yaml

from heavy_prefill_bench.runner import run_autotune


async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    await run_autotune(config)


if __name__ == "__main__":
    asyncio.run(main())
