"""vLLM server launcher and adapter."""
import asyncio
import subprocess
import time
from typing import Any, Dict

import aiohttp

from heavy_prefill_bench.adapters.openai_adapter import OpenAIAdapter


class VLLMAdapter(OpenAIAdapter):
    """Launch vLLM serve and communicate via OpenAI-compatible HTTP API."""

    def __init__(
        self,
        model: str,
        port: int = 8000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        enable_chunked_prefill: bool = True,
        trust_remote_code: bool = True,
        dtype: str = "auto",
    ):
        super().__init__(base_url=f"http://localhost:{port}/v1")
        self.model = model
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.enable_chunked_prefill = enable_chunked_prefill
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self._process: subprocess.Popen | None = None

    @property
    def name(self) -> str:
        return "vllm"

    def start(self, config: Dict[str, Any]) -> None:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
        ]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        if self.max_num_batched_tokens is not None:
            cmd.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if self.max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(self.max_num_seqs)])

        print(f"[vLLM] Starting server: {' '.join(cmd)}")
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for server readiness
        self._wait_for_ready(timeout_sec=300)
        super().start(config)
        print("[vLLM] Server ready.")

    def _wait_for_ready(self, timeout_sec: float = 300.0):
        url = f"http://localhost:{self.port}/health"
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1.0)
            # Print a dot to show progress
            print(".", end="", flush=True)
        raise RuntimeError("vLLM server failed to start within timeout")

    def stop(self) -> None:
        super().stop()
        if self._process:
            print("[vLLM] Shutting down server...")
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
