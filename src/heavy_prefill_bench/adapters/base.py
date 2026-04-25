"""Abstract adapter interface for inference frameworks."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class FrameworkAdapter(ABC):
    """Base class for framework-specific launch and request handling."""

    @abstractmethod
    async def start(self, config: Dict[str, Any]) -> None:
        """Start the inference server or load the model."""
        pass

    @abstractmethod
    async def send_request(
        self,
        prompt_token_ids: List[int],
        max_tokens: int,
        min_tokens: int,
        ignore_eos: bool = True,
    ) -> Dict[str, Any]:
        """Send a single request and return timing + token metadata."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Shut down the server / release resources."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
