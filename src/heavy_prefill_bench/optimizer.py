"""SGLang bench_offline_throughput sweep driver.

Runs sglang.bench_offline_throughput for each chunked_prefill_size in the sweep list,
parses the JSONL output, and finds the configuration with maximum throughput.
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List


def _detect_gpu() -> str:
    """Auto-detect GPU name(s) via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise RuntimeError(f"GPU auto-detection failed: {exc}")

    names = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not names:
        raise RuntimeError("GPU auto-detection returned no GPUs")

    unique = sorted(set(names))
    if len(unique) == 1:
        return unique[0]
    counts = {name: names.count(name) for name in unique}
    return ", ".join(
        f"{count}x{name}" if count > 1 else name for name in unique
    )


def _build_command(
    model: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    chunk_size: int,
    output_file: str,
    tp: int = 1,
    mem_fraction_static: float = 0.85,
    random_range_ratio: float = 0.0,
    quantization: str | None = None,
) -> List[str]:
    cmd = [
        sys.executable, "-m", "sglang.bench_offline_throughput",
        "--model-path", model,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--random-range-ratio", str(random_range_ratio),
        "--num-prompts", str(num_prompts),
        "--tensor-parallel-size", str(tp),
        "--mem-fraction-static", str(mem_fraction_static),
        "--disable-radix-cache",
        "--chunked-prefill-size", str(chunk_size),
        "--result-filename", output_file,
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])
    return cmd


def _parse_jsonl(output_file: str) -> Dict[str, Any] | None:
    """Parse the LAST throughput record from SGLang JSONL output."""
    if not os.path.exists(output_file):
        return None
    last = None
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "request_throughput" in record:
                last = record
    return last


class AutoTuner:
    """Sweeps chunked_prefill_size via SGLang bench_offline_throughput."""

    def __init__(self, config: Dict[str, Any]):
        self.model = config["model"]
        wl = config["workload"]
        self.input_len = wl["input_len"]
        self.output_len = wl["output_len"]
        self.num_prompts = wl["num_prompts"]

        sweep = config["sweep"]
        self.chunk_sizes = list(sweep["chunked_prefill_sizes"])

        sga = config.get("sglang_args", {})
        self.tp = sga.get("tp", 1)
        self.mem_fraction_static = sga.get("mem_fraction_static", 0.85)
        self.random_range_ratio = sga.get("random_range_ratio", 0.0)
        self.quantization = sga.get("quantization") or None
        self.extra_args = sga.get("extra_args", [])

        self.output_dir = config.get("output_dir", "results")
        self.framework = config.get("framework", "sglang")

        self.gpu_label = _detect_gpu()
        self.gpu_hourly_cost_usd = config["hardware"]["gpu_hourly_cost_usd"]

    def run(self) -> List[Dict[str, Any]]:
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[AutoTuner] Model: {self.model}")
        print(f"[AutoTuner] Workload: input={self.input_len}, output={self.output_len}, "
              f"prompts={self.num_prompts}")
        print(f"[AutoTuner] Sweep: chunked_prefill_size={self.chunk_sizes}")
        print(f"[AutoTuner] TP={self.tp}, mem_fraction_static={self.mem_fraction_static}")
        print(f"[AutoTuner] GPU: {self.gpu_label}")

        results: List[Dict[str, Any]] = []

        for i, chunk_size in enumerate(self.chunk_sizes):
            output_file = os.path.join(
                self.output_dir, f"sglang_chunk{chunk_size}.jsonl"
            )
            cmd = _build_command(
                model=self.model,
                input_len=self.input_len,
                output_len=self.output_len,
                num_prompts=self.num_prompts,
                chunk_size=chunk_size,
                output_file=output_file,
                tp=self.tp,
                mem_fraction_static=self.mem_fraction_static,
                random_range_ratio=self.random_range_ratio,
                quantization=self.quantization,
            )
            if self.extra_args:
                cmd.extend(self.extra_args)

            print(f"\n[{i + 1}/{len(self.chunk_sizes)}] chunked_prefill_size={chunk_size}")
            print(f"  {' '.join(cmd)}")
            sys.stdout.flush()

            try:
                subprocess.run(cmd, check=True, timeout=7200)
            except subprocess.CalledProcessError as exc:
                print(f"  FAILED (exit={exc.returncode}) — likely OOM or config error")
                continue
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT — exceeded 2h")
                continue

            parsed = _parse_jsonl(output_file)
            if parsed is None:
                print(f"  No throughput data in output file")
                continue

            result = {
                "chunked_prefill_size": chunk_size,
                "num_prompts": self.num_prompts,
                "input_len": self.input_len,
                "output_len": self.output_len,
                "requests_per_sec": parsed.get("request_throughput", 0),
                "input_tokens_per_sec": parsed.get("input_throughput", 0),
                "output_tokens_per_sec": parsed.get("output_throughput", 0),
                "total_tokens_per_sec": parsed.get("total_throughput", 0),
                "requests_per_hour": parsed.get("request_throughput", 0) * 3600,
                "successful_requests": parsed.get("successful_requests", 0),
                "total_output_tokens": parsed.get("total_output_tokens", 0),
                "model": self.model,
                "tp": self.tp,
                "framework": self.framework,
                "gpu": self.gpu_label,
                "quantization": self.quantization or "bf16",
            }
            results.append(result)

            print(f"  requests/sec: {result['requests_per_sec']:.3f}  "
                  f"tokens/sec: {result['total_tokens_per_sec']:.0f}  "
                  f"→ {result['requests_per_hour']:.0f} req/hr")

        if results:
            best = max(results, key=lambda r: r["requests_per_sec"])
            print(f"\n{'=' * 60}")
            print(f"[AutoTuner] BEST CONFIG:")
            print(f"  chunked_prefill_size = {best['chunked_prefill_size']}")
            print(f"  requests/hour        = {best['requests_per_hour']:.0f}")
            print(f"  tokens/sec           = {best['total_tokens_per_sec']:.0f}")
            print(f"{'=' * 60}")
        else:
            print("\n[AutoTuner] No successful runs. Check model path and VRAM.")

        return results
