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
│   └── reporter.py                    # CSV writer for sweep results
└── results/                           # Output CSVs, JSONL, metadata
```

The auto-tuner uses SGLang's built-in `bench_offline_throughput` which runs the Engine in-process with no HTTP overhead — the correct tool for pure throughput measurement.

GPU name is **auto-detected** via `nvidia-smi` and embedded in every output row so cross-machine comparisons are never mislabeled.

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (SGLang)
uv pip install -e ".[sglang]"
```

## Configuration

Edit `config.yaml`:

```yaml
model: microsoft/Phi-4-mini-instruct

workload:
  input_len: 4000
  output_len: 1000

sweep:
  num_prompts: [50, 100, 200, 400]
  chunked_prefill_sizes: [2048, 4096, 8192, 16384, 32768]

sglang_args:
  tp: 1
  mem_fraction_static: 0.85
  disable_radix_cache: true
  random_range_ratio: 0.0
  quantization: null

hardware:
  gpu_hourly_cost_usd: 0.34
  provider_name: runpod
  instance_type: 1x RTX 4090
  region: us-ca
  pricing_timestamp_utc: "2026-04-26T00:00:00+00:00"
  pricing_source_note: "RunPod on-demand listed price"

output_dir: results
```

The optimizer runs `python -m sglang.bench_offline_throughput` for each `(num_prompts, chunked_prefill_size)` combination, parses the JSONL output, and picks the best. The sweep stops early when a chunk size OOMs (larger chunks will also OOM) or when throughput plateaus across num_prompts levels (GPU is saturated).

| Field | Meaning |
|---|---|
| `sweep.num_prompts` | `[50, 100, 200, 400]` = sweep batch sizes to find GPU saturation point. The sweep stops early when throughput plateaus (<2% gain) or all chunk sizes OOM. Cannot be used together with `workload.num_prompts`. |
| `workload.num_prompts` | Fixed batch size (single value). Cannot be used together with `sweep.num_prompts`. |
| `random_range_ratio` | `0.0` = all prompts exactly `input_len`. `1.0` = uniform 0–2× input_len. Use `0.0` for deterministic batch jobs. |
| `quantization` | `null` = bf16 on Ampere/Ada+ GPUs, fp16 on older ones. Other options: `fp8`, `awq`, `gptq`. Only applied if non-null. |
| `gpu_hourly_cost_usd` | **Required.** Used to compute `tokens_per_dollar`. |
| `provider_name` | Provider label persisted in metadata pricing provenance. |
| `instance_type` | Instance SKU/type persisted in metadata pricing provenance. |
| `region` | Region persisted in metadata pricing provenance. |
| `pricing_timestamp_utc` | Optional pricing snapshot timestamp; defaults to current UTC time. |
| `pricing_source_note` | Optional note (pricing page/source) for auditability. |

## Running

```bash
source .venv/bin/activate
python run_benchmark.py config.yaml
```

Each sweep config takes ~2–3 minutes on RTX 4090 (model load ~45s, benchmark ~100s for 50×4k/1k). A 5-config sweep finishes in ~12–15 minutes.

## Output

One CSV at `results/sglang_autotune.csv` with sweep-level schema:

```
framework, gpu, quantization, chunked_prefill_size, num_prompts, input_len, output_len,
requests_per_sec, input_tokens_per_sec, output_tokens_per_sec,
total_tokens_per_sec, requests_per_hour, successful_requests,
total_output_tokens, model, tp, gpu_hourly_cost_usd, tokens_per_dollar, oom
```

Plus `results/sglang_autotune_metadata.json` with run configuration and explicit pricing provenance (`provider_name`, `instance_type`, `region`, `pricing_timestamp_utc`, `pricing_source_note`), and per-config JSONL files at `results/sglang_p{num_prompts}_chunk{chunk_size}.jsonl`.

After each run, the tool warns if the output directory contains metadata files with mixed pricing provenance. Treat that warning as a blocker for direct tokens-per-dollar comparisons.

## CSV comparison rules (important)

When merging or comparing benchmark CSVs, only compare `tokens_per_dollar` when all of the following match:

1. `provider_name`
2. `instance_type`
3. `region`
4. `gpu_hourly_cost_usd`
5. `pricing_source_note` and `pricing_timestamp_utc` (same pricing snapshot/provenance)

If any differ, do **not** rank by `tokens_per_dollar` across those rows. You may still compare raw throughput (`requests_per_sec`, `total_tokens_per_sec`) for model/topology behavior.

## Metrics

| Metric | Definition |
|---|---|
| Request throughput | `requests/sec` — primary metric. Multiply by 3600 for req/hr. |
| Input token throughput | `input_tokens/sec` — prefill throughput |
| Output token throughput | `output_tokens/sec` — decode throughput |
| Total token throughput | `total_tokens/sec` — input + output |
| Tokens per dollar | `total_tokens/sec × 3600 / gpu_hourly_cost_usd` — cost-normalized throughput for GPU selection |

> **Note:** Configurations that OOM or fail appear in the CSV with throughput fields empty and `oom: True`. These rows are excluded from best-config selection but preserved so you can see where each GPU saturates.

## Example Results

RTX 4090 (24 GB), Qwen2.5-7B-Instruct, bf16, workload 4k input × 1k output × 50 prompts, $0.70/hr:

| chunked_prefill_size | req/sec | req/hr | tokens/sec | tokens/$ |
|---|---|---|---|---|
| 2048 | 1.63 | 5,851 | 4,137 | 21,274,240 |
| 4096 | 1.63 | 5,881 | 4,158 | 21,385,047 |
| 8192 | 1.64 | 5,892 | 4,166 | 21,426,423 |
| 16384 | — | — | — | OOM |
| 32768 | — | — | — | OOM |

> **Note:** Larger `chunked_prefill_size` values OOM on 24 GB because they process more tokens simultaneously during prefill. The sweet spot is the largest size that fits.

Older result — RTX 4090 (24 GB), Phi-4-mini-instruct (3.8B), workload 4k input × 1k output × 50 prompts, $0.34/hr:

| chunked_prefill_size | req/sec | req/hr | tokens/sec | tokens/$ |
|---|---|---|---|---|
| 2048 | 2.00 | 7,211 | 5,098 | 53,977,412 |
| 4096 | 2.01 | 7,241 | 5,120 | 54,211,765 |
| 8192 | 2.01 | 7,241 | 5,120 | 54,211,765 |

### H100 80GB (RunPod, $2.99/hr)

RunPod H100 SXM (80 GB HBM3), workload 4k input × 1k output × 50 prompts, chunked_prefill_sizes: [2048, 4096, 8192, 16384, 32768].

> **Session note (2026-04-27):** The latest sweep with the new auto-tuner code (commit `4b3c622`) completed 19/20 configs for 7B before hitting a 1-hour bash timeout. 14B and 32B were not started. See [`SESSION_2026-04-27.md`](SESSION_2026-04-27.md) for full details and how to finish the run.

**Qwen2.5-7B-Instruct (bf16)** — *19/20 configs completed*

| num_prompts | chunked_prefill_size | req/sec | req/hr | tokens/sec | tokens/$ |
|---|---:|---:|---:|---:|---:|
| 50 | 2048 | 4.85 | 17,476 | 12,356 | 14,876,908 |
| 50 | 4096 | 5.01 | 18,028 | 12,747 | 15,347,081 |
| 50 | 8192 | 5.09 | 18,322 | 12,955 | 15,597,870 |
| 50 | 16384 | 5.05 | 18,170 | 12,847 | 15,467,793 |
| 50 | 32768 | 5.07 | 18,234 | 12,893 | 15,522,963 |
| 100 | 2048 | 7.38 | 26,555 | 18,773 | 22,602,851 |
| 100 | 4096 | 7.41 | 26,671 | 18,855 | 22,702,145 |
| 100 | 8192 | 7.47 | 26,888 | 19,008 | 22,886,232 |
| 100 | 16384 | 7.50 | 27,002 | 19,089 | 22,983,993 |
| 100 | 32768 | 7.53 | 27,102 | 19,160 | 23,068,902 |
| 200 | 2048 | 9.74 | 35,077 | 24,636 | 29,662,113 |
| 200 | 4096 | 9.82 | 35,353 | 24,830 | 29,895,737 |
| 200 | 8192 | 9.95 | 35,812 | 25,152 | 30,283,507 |
| 200 | 16384 | 10.00 | 35,989 | 25,277 | 30,433,840 |
| 200 | 32768 | 10.06 | 36,225 | 25,443 | 30,633,138 |
| 400 | 2048 | 11.02 | 39,672 | 28,056 | 33,779,796 |
| 400 | 4096 | 11.11 | 39,983 | 28,276 | 34,044,546 |
| 400 | 8192 | 11.28 | 40,608 | 28,718 | 34,576,808 |
| 400 | 16384 | 11.32 | 40,769 | 28,832 | 34,714,497 |
| 400 | 32768 | — | — | — | **missing** |

> **Note:** `p400_chunk32768` timed out and was not recorded. All other 19 configs succeeded with no OOMs.

**Qwen2.5-14B-Instruct (bf16)** — *prior complete run*

| chunked_prefill_size | req/sec | req/hr | tokens/sec | tokens/$ |
|---|---|---|---|---|
| 2048 | 2.44 | 8,786 | 6,212 | 7,479,311 |
| 4096 | 2.46 | 8,847 | 6,255 | 7,531,468 |
| 8192 | 2.47 | 8,897 | 6,290 | 7,573,664 |
| 16384 | 2.47 | 8,875 | 6,275 | 7,555,093 |
| 32768 | 2.48 | 8,926 | 6,311 | 7,598,311 |

**Qwen2.5-32B-Instruct (fp8)** — *prior complete run*

| chunked_prefill_size | req/sec | req/hr | tokens/sec | tokens/$ |
|---|---|---|---|---|
| 2048 | 1.83 | 6,594 | 4,662 | 5,613,335 |
| 4096 | 1.81 | 6,503 | 4,598 | 5,536,168 |
| 8192 | 1.79 | 6,431 | 4,547 | 5,474,720 |
| 16384 | 1.74 | 6,260 | 4,426 | 5,329,371 |
| 32768 | 1.70 | 6,125 | 4,331 | 5,214,092 |

> **Note:** 32B fp8 throughput *decreases* with larger `chunked_prefill_size` on this workload, unlike 7B and 14B where larger chunks improved throughput. The sweet spot varies by model size and quantization.

Raw CSVs: [`results/sglang_autotune.csv`](results/sglang_autotune.csv) (current, 19/20 rows), [`results/sglang_autotune_metadata.json`](results/sglang_autotune_metadata.json). Prior runs: [`results/sglang_autotune_Qwen2.5-7B-bf16.csv`](results/sglang_autotune_Qwen2.5-7B-bf16.csv), [`results/sglang_autotune_Qwen2.5-14B-bf16.csv`](results/sglang_autotune_Qwen2.5-14B-bf16.csv), [`results/sglang_autotune_Qwen2.5-32B-fp8.csv`](results/sglang_autotune_Qwen2.5-32B-fp8.csv), [`results/all_runs.csv`](results/all_runs.csv).

## Troubleshooting

### `ImportError: libnuma.so.1: cannot open shared object file`

SGLang's `sgl_kernel` requires `libnuma1`. If you see this error when the benchmark subprocess starts:

```
ImportError: libnuma.so.1: cannot open shared object file: No such file or directory
```

Install it with:

```bash
apt-get update && apt-get install -y libnuma1
```

Then re-run the benchmark.

### `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`

SGLang's FlashInfer backend JIT-compiles CUDA kernels during graph capture and requires `ninja-build`:

```bash
apt-get update && apt-get install -y ninja-build
```

### Disk quota exceeded during model download

The HuggingFace cache defaults to `~/.cache/huggingface`, which may be on a small root partition. Move it to a larger mount (e.g. `/workspace` on RunPod) and symlink back:

```bash
mv ~/.cache/huggingface /workspace/huggingface_cache
ln -s /workspace/huggingface_cache ~/.cache/huggingface
```

### Bash / tool timeout on long sweeps

A full 20-config sweep (4 batch sizes × 5 chunk sizes) can take 45–60 minutes on H100. If the runner kills the process at the 1-hour mark, the Python script may be interrupted before it writes `sglang_autotune.csv` and `sglang_autotune_metadata.json`.

**Fix:**
- Use a timeout longer than 60 minutes if your runner supports it.
- Or run one model at a time and back up results between models.
- If interrupted, existing `results/sglang_p{num_prompts}_chunk{chunk_size}.jsonl` files are still valid. Parse them directly rather than re-running the whole sweep.

### `pip: command not found` inside `.venv`

If the virtual environment was created with `uv`, `pip` is not installed as a standalone binary.

**Fix:** Use `python -m pip` or `uv pip`:
```bash
python -m pip uninstall -y hf-xet
# or
uv pip uninstall hf-xet
```
