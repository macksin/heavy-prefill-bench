"""OpenAI-compatible HTTP adapter (works for both vLLM and SGLang servers)."""
import asyncio
import time
from typing import Any, Dict, List

import aiohttp

from heavy_prefill_bench.adapters.base import FrameworkAdapter


class OpenAIAdapter(FrameworkAdapter):
    """Generic adapter for OpenAI-compatible HTTP endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "dummy"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "openai"

    async def start(self, config: Dict[str, Any]) -> None:
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

    async def send_request(
        self,
        prompt_token_ids: List[int],
        max_tokens: int,
        min_tokens: int,
        ignore_eos: bool = True,
    ) -> Dict[str, Any]:
        if self.session is None:
            raise RuntimeError("Adapter not started")

        url = f"{self.base_url}/completions"
        payload = {
            "model": "default",
            "prompt_token_ids": prompt_token_ids,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "ignore_eos": ignore_eos,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0,
        }

        t_start = time.perf_counter()
        t_first = None
        chunks = 0
        output_text = ""
        usage = None

        async with self.session.post(url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                import json

                event = json.loads(data)
                choices = event.get("choices", [])
                if choices:
                    delta = choices[0].get("text", "")
                    if delta:
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks += 1
                        output_text += delta
                # Final chunk may contain usage
                if "usage" in event and event["usage"]:
                    usage = event["usage"]

        t_end = time.perf_counter()
        ttft_ms = (t_first - t_start) * 1000 if t_first else 0.0
        e2e_ms = (t_end - t_start) * 1000
        output_tokens = usage.get("completion_tokens", 0) if usage else 0
        tpot_ms = (e2e_ms - ttft_ms) / output_tokens if output_tokens > 0 else 0.0

        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "output_tokens": output_tokens,
            "chunks": chunks,
        }

    async def stop(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None
