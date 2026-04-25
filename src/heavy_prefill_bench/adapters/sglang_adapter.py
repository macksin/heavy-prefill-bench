"""SGLang server launcher and adapter."""
import subprocess
import time
from typing import Any, Dict

from heavy_prefill_bench.adapters.openai_adapter import OpenAIAdapter


class SGLangAdapter(OpenAIAdapter):
    """Launch SGLang server and communicate via OpenAI-compatible HTTP API."""

    def __init__(
        self,
        model: str,
        port: int = 8001,
        tp_size: int = 1,
        chunked_prefill_size: int | None = None,
        max_running_requests: int | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        super().__init__(base_url=f"http://localhost:{port}/v1")
        self.model = model
        self.port = port
        self.tp_size = tp_size
        self.chunked_prefill_size = chunked_prefill_size
        self.max_running_requests = max_running_requests
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._process: subprocess.Popen | None = None

    @property
    def name(self) -> str:
        return "sglang"

    async def start(self, config: Dict[str, Any]) -> None:
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model,
            "--port", str(self.port),
            "--tp", str(self.tp_size),
            "--dtype", self.dtype,
        ]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.chunked_prefill_size is not None:
            cmd.extend(["--chunked-prefill-size", str(self.chunked_prefill_size)])
        if self.max_running_requests is not None:
            cmd.extend(["--max-running-requests", str(self.max_running_requests)])

        print(f"[SGLang] Starting server: {' '.join(cmd)}")
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self._wait_for_ready(timeout_sec=300)
        await super().start(config)
        print("[SGLang] Server ready.")

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
            print(".", end="", flush=True)
        raise RuntimeError("SGLang server failed to start within timeout")

    async def stop(self) -> None:
        await super().stop()
        if self._process:
            print("[SGLang] Shutting down server...")
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
