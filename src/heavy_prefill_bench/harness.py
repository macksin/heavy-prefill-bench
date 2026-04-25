"""Async client measurement harness."""
import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any, List

from heavy_prefill_bench.adapters.base import FrameworkAdapter
from heavy_prefill_bench.generator import generate_request_tokens
from heavy_prefill_bench.monitor import GPUMonitor


@dataclass
class RequestResult:
    request_id: int
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    chunks: int
    error: str | None = None


@dataclass
class BenchmarkResult:
    request_results: List[RequestResult] = field(default_factory=list)
    wall_time_sec: float = 0.0
    gpu_metrics: Any | None = None
    framework: str = ""
    chunk_size: int | None = None
    max_seqs: int | None = None
    concurrency: int = 0


class Harness:
    """Framework-agnostic async benchmark harness."""

    def __init__(
        self,
        adapter: FrameworkAdapter,
        concurrency: int,
        gpu_monitor: GPUMonitor | None = None,
    ):
        self.adapter = adapter
        self.concurrency = concurrency
        self.gpu_monitor = gpu_monitor

    async def run(
        self,
        prompts: List[List[int]],
        output_len: int = 2000,
        ignore_eos: bool = True,
    ) -> BenchmarkResult:
        semaphore = asyncio.Semaphore(self.concurrency)

        if self.gpu_monitor:
            self.gpu_monitor.start()

        # Warmup: 1 short request to trigger torch.compile + CUDA graph capture
        warmup_prompt = generate_request_tokens(100, seed=999999)
        print("[Harness] Running warmup (1×100 tokens)...")
        t_warmup = time.perf_counter()
        try:
            await self.adapter.send_request(
                prompt_token_ids=warmup_prompt,
                max_tokens=5,
                min_tokens=5,
                ignore_eos=True,
            )
        except Exception as exc:
            print(f"[Harness] Warmup failed: {exc}", file=sys.stderr)
        warmup_sec = time.perf_counter() - t_warmup
        print(f"[Harness] Warmup complete ({warmup_sec:.1f}s). Starting benchmark...")

        wall_start = time.perf_counter()

        async def _send_one(idx: int, prompt: List[int]) -> RequestResult:
            async with semaphore:
                try:
                    resp = await self.adapter.send_request(
                        prompt_token_ids=prompt,
                        max_tokens=output_len,
                        min_tokens=output_len,
                        ignore_eos=ignore_eos,
                    )
                    return RequestResult(
                        request_id=idx,
                        input_tokens=len(prompt),
                        output_tokens=resp["output_tokens"],
                        ttft_ms=resp["ttft_ms"],
                        tpot_ms=resp["tpot_ms"],
                        e2e_ms=resp["e2e_ms"],
                        chunks=resp["chunks"],
                    )
                except Exception as exc:
                    print(f"[Harness] Request {idx} failed: {exc}", file=sys.stderr)
                    return RequestResult(
                        request_id=idx,
                        input_tokens=len(prompt),
                        output_tokens=0,
                        ttft_ms=0.0,
                        tpot_ms=0.0,
                        e2e_ms=0.0,
                        chunks=0,
                        error=str(exc),
                    )

        tasks = [_send_one(i, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

        wall_end = time.perf_counter()
        wall_time_sec = wall_end - wall_start

        # Error threshold check: if >50% of requests fail, raise — do not write zero CSV
        failures = [r for r in results if r.error is not None]
        if len(failures) > len(prompts) * 0.5:
            error_summary = "\n".join(f"  req={r.request_id}: {r.error}" for r in failures)
            raise RuntimeError(
                f"Too many failures: {len(failures)}/{len(prompts)} requests failed.\n{error_summary}"
            )

        gpu_metrics = None
        if self.gpu_monitor:
            gpu_metrics = self.gpu_monitor.stop()

        return BenchmarkResult(
            request_results=results,
            wall_time_sec=wall_time_sec,
            gpu_metrics=gpu_metrics,
            framework=self.adapter.name,
            concurrency=self.concurrency,
        )
