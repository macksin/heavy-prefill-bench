# heavy-prefill-bench

Benchmarking suite for prefill-heavy LLM inference workloads (50k input / 2k output) across vLLM, SGLang, and TensorRT-LLM.

## Why This Benchmark?

A 25:1 prefill-to-decode ratio exposes bottlenecks that standard benchmarks (512–2k input) miss entirely:
- Prefill dominates ~90% of total latency
- Chunked prefill configuration is make-or-break
- KV cache allocation per sequence is enormous, reducing effective batch size
- Framework memory schedulers matter more than decode kernels

## Architecture

```
heavy-prefill-bench/
├── config.yaml                        # Sweep parameters and model config
├── run_benchmark.py                   # Main entry point
├── pyproject.toml                     # uv-managed dependencies
├── situation.txt                      # Detailed situation report & correction spec
├── src/heavy_prefill_bench/
│   ├── generator.py                   # Synthetic token generation (per-request seeds)
│   ├── harness.py                     # Async benchmark harness (semaphore concurrency)
│   ├── monitor.py                     # GPU util/VRAM background sampler (nvidia-smi)
│   ├── reporter.py                    # CSV writer with fixed schema
│   ├── runner.py                      # Sweep orchestrator (chunk_size × max_seqs × concurrency)
│   └── adapters/
│       ├── base.py                    # Abstract FrameworkAdapter interface
│       ├── vllm_async_adapter.py      # vLLM in-process adapter (AsyncLLMEngine)
│       ├── openai_adapter.py          # OpenAI-compatible HTTP adapter (for SGLang)
│       └── sglang_adapter.py          # SGLang server launcher + HTTP adapter
└── results/                           # Output CSVs and metadata
```

### Adapter Strategy

- **vLLM**: Uses `AsyncLLMEngine` (in-process, no HTTP). This avoids the V1 engine subprocess crash observed on vLLM 0.19.1 and eliminates HTTP overhead for the most accurate timing.
- **SGLang**: Uses HTTP via OpenAI-compatible API (SGLang has no in-process Python API).
- **TensorRT-LLM / MLC-LLM**: Stub adapters for future implementation.

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[vllm]"      # For vLLM benchmarks
uv pip install -e ".[sglang]"     # For SGLang benchmarks
uv pip install -e ".[all]"        # For both
```

## Configuration

Edit `config.yaml`:

```yaml
model: microsoft/Phi-4-mini-instruct
max_model_len: 52000          # Must be >= input_len + output_len
input_len: 50000
output_len: 2000
num_requests: 100
chunk_sizes: [4096, 8192]
max_seqs: [4, 8, 16]
concurrencies: [1, 2, 4, 8, 16]
output_dir: results
frameworks:
  - vllm
  # - sglang
```

### VRAM Budget (RTX 4090, 24 GB, Phi-4-mini)

| Component | Size |
|---|---|
| Model weights (bf16) | ~7.2 GB |
| CUDA overhead | ~0.5 GB |
| torch.compile / CUDA graphs | ~0.5–1.0 GB |
| **Available for KV cache** | **~13–14 GB** |
| KV per token (32 layers, 8 KV heads, head_dim 128, bf16) | 128 KB |
| KV per 52k-token sequence | ~6.4 GB |
| **Max concurrent sequences at 52k** | **~2** |

At `max_model_len=52000`, only 2 sequences can fit in KV cache concurrently. The concurrency sweep reveals the point where vLLM queues requests and latency explodes.

## Running

```bash
# Activate environment
source .venv/bin/activate

# Run full benchmark
python run_benchmark.py config.yaml
```

## Output

One CSV per framework with the fixed schema:

```
request_id, input_tokens, output_tokens, ttft_ms, tpot_ms, e2e_ms,
gpu_util_mean, vram_peak_gb, framework, chunk_size, max_seqs, concurrency
```

Plus a companion `metadata.json` per run with all configuration parameters.

## Metrics

| Metric | Definition |
|---|---|
| TTFT | Time to First Token (ms) — measures prefill cost |
| TPOT | Time Per Output Token (ms) — measures decode throughput |
| E2E | End-to-end latency (ms) — total request time |
| Prefill throughput | `total_input_tokens / sum(TTFTs)` (tokens/sec) |
| Decode throughput | `total_output_tokens / decode_time` (tokens/sec) |
| Job wall time | Total time to process N requests |

Distributions collected: p50, p90, p95, p99 of TTFT, TPOT, E2E.

## Phase 1: Validation (Current)

- **Hardware**: 1× RTX 4090 (24 GB)
- **Model**: microsoft/Phi-4-mini-instruct (3.8B, 128k context, 7.67 GB bf16)
- **Objective**: Validate measurement harness, verify cross-framework metric parity, confirm synthetic generator correctness

## Phase 2: Real Benchmark (Future)

- **Hardware**: 4× or 8× H100 SXM
- **Model**: DeepSeek V4 Flash (or similar large model)
- **Objective**: Full sweep of chunk_size × max_seqs × concurrency across frameworks

## Known Issues

See `situation.txt` for the full detailed situation report. Summary:

1. **vLLM server mode crashes** on vLLM 0.19.1 — V1 EngineCore subprocess becomes zombie. Fixed by using `AsyncLLMEngine` in-process adapter.
2. **vLLM defaults to max_model_len=4096** for Phi-4-mini (reads `original_max_position_embeddings` instead of `max_position_embeddings=131072`). Must set `max_model_len` explicitly.
3. **SGLang has no native Phi-4-mini model file** — may load via `--trust-remote-code` and `phi3.py`, but untested. Fallback: Gemma-3-4b-it.
4. **Async lifecycle bug** in original HTTP adapter — `session.close()` called from running event loop. Fixed: made `start()`/`stop()` async.