"""vLLM AsyncLLMEngine in-process adapter."""
import time
import uuid
from typing import Any, Dict, List

from heavy_prefill_bench.adapters.base import FrameworkAdapter


class VLLMAsyncAdapter(FrameworkAdapter):
    """vLLM benchmark adapter using AsyncLLMEngine in-process.

    Avoids the V1 EngineCore subprocess zombie issue seen with server mode.
    Uses streaming output from engine.generate() for TTFT/TPOT measurement.
    """

    def __init__(
        self,
        model: str,
        max_model_len: int = 52000,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
    ):
        self.model = model
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self._engine = None

    @property
    def name(self) -> str:
        return "vllm"

    async def start(self, config: Dict[str, Any]) -> None:
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        max_batched = self.max_num_batched_tokens or 8192
        max_seqs = self.max_num_seqs

        engine_args = AsyncEngineArgs(
            model=self.model,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=max_batched,
            max_num_seqs=max_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            disable_log_stats=True,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        print(f"[vLLM] Engine created (max_model_len={self.max_model_len}, "
              f"max_num_batched_tokens={max_batched}, max_num_seqs={max_seqs})")

    async def send_request(
        self,
        prompt_token_ids: List[int],
        max_tokens: int,
        min_tokens: int,
        ignore_eos: bool = True,
    ) -> Dict[str, Any]:
        if self._engine is None:
            raise RuntimeError("Adapter not started")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            ignore_eos=ignore_eos,
            temperature=0.0,
        )

        t_start = time.perf_counter()
        t_first = None
        output_tokens = 0

        request_id = str(uuid.uuid4())
        async for output in self._engine.generate(
            prompt=prompt_token_ids,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if t_first is None:
                t_first = time.perf_counter()
            if output.outputs:
                output_tokens = len(output.outputs[0].token_ids)

        t_end = time.perf_counter()
        ttft_ms = (t_first - t_start) * 1000 if t_first else 0.0
        e2e_ms = (t_end - t_start) * 1000
        tpot_ms = (e2e_ms - ttft_ms) / output_tokens if output_tokens > 0 else 0.0

        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "output_tokens": output_tokens,
            "chunks": 0,
        }

    async def stop(self) -> None:
        if self._engine:
            try:
                self._engine.shutdown()
            except Exception as exc:
                print(f"[vLLM] Warning during engine shutdown: {exc}")
            self._engine = None
