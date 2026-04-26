# heavy-prefill-bench

Automatic topology optimizer for maximum throughput on a fixed workload shape. Sweeps SGLang's `bench_offline_throughput` and finds the best `chunked_prefill_size` for maximum prompts/hour.

## Architecture

```
heavy-prefill-bench/
├── config.yaml                        # Model, workload shape, sweep params
├── run_benchmark.py                   # CLI entry point
├── pyproject.toml                     # Dependencies
├── src/heavy_prefill_bench/
│   ├── optimizer.py                   # Auto-tuner: subprocess driver + JSONL parser
│   ├── runner.py                      # Config validation and glue
│   ├── reporter.py                    # CSV writer for sweep results
│   ├── metrics.py                     # Throughput metrics
│   ├── harness.py                     # (legacy vLLM path — unused)
│   ├── generator.py                   # (legacy — unused)
│   ├── monitor.py                     # (legacy — unused)
│   └── adapters/                      # (legacy — unused)
└── results/                           # Output CSVs, JSONL, metadata
```

The auto-tuner uses SGLang's built-in `bench_offline_throughput` which runs the Engine in-process with no HTTP overhead — the correct tool for pure throughput measurement.

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (SGLang)
uv pip install -e ".[sglang]"

# Optional: install libnuma for sgl-kernel on some systems
apt-get install -y libnuma1
```

## Configuration

Edit `config.yaml`:

```yaml
model: microsoft/Phi-4-mini-instruct

workload:
  input_len: 2000
  output_len: 1000
  num_prompts: 200

sweep:
  chunked_prefill_sizes: [2048, 4096, 8192]

sglang_args:
  tp: 1
  mem_fraction_static: 0.85
  disable_radix_cache: true
  random_range_ratio: 1.0

output_dir: results
```

The optimizer runs `python -m sglang.bench_offline_throughput` for each `chunked_prefill_size` in the sweep list, parses the JSONL output, and picks the best.

## Running

```bash
source .venv/bin/activate
python run_benchmark.py config.yaml
```

Each sweep config takes ~4 minutes on RTX 4090 (model load ~45s, benchmark ~165s for 200×2k/1k).

## Output

One CSV at `results/sglang_autotune.csv` with sweep-level schema:

```
framework, chunked_prefill_size, num_prompts, input_len, output_len,
requests_per_sec, input_tokens_per_sec, output_tokens_per_sec,
total_tokens_per_sec, requests_per_hour, successful_requests,
total_output_tokens, model, tp
```

Plus `results/sglang_autotune_metadata.json` with run configuration, and per-config JSONL files at `results/sglang_chunk{size}.jsonl`.

## Metrics

| Metric | Definition |
|---|---|
| Request throughput | `requests/sec` — primary metric. Multiply by 3600 for req/hr. |
| Input token throughput | `input_tokens/sec` — prefill throughput |
| Output token throughput | `output_tokens/sec` — decode throughput |
| Total token throughput | `total_tokens/sec` — input + output |

## Example Results

RTX 4090 (24 GB), Phi-4-mini-instruct (3.8B), workload 2k input × 1k output × 200 prompts:

| chunked_prefill_size | req/sec | req/hr | tokens/sec |
|---|---|---|---|
| 2048 | 1.21 | 4,349 | 3,624 |
| 4096 | 1.21 | 4,359 | 3,632 |
| **8192** | **1.22** | **4,380** | **3,650** |
