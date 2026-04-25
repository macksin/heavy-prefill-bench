"""Synthetic token generator for benchmark inputs."""
import numpy as np
from typing import List

VOCAB_SIZE = 200_064  # Phi-4-mini vocabulary size
SAFE_TOKEN_MIN = 1000
SAFE_TOKEN_MAX = VOCAB_SIZE - 100


def generate_request_tokens(input_len: int = 50_000, seed: int | None = None) -> List[int]:
    """Generate a single request's input token IDs.

    Uses a per-request seed to ensure different inputs per request
    (disabling accidental prefix caching).
    """
    rng = np.random.default_rng(seed)
    tokens = rng.integers(SAFE_TOKEN_MIN, SAFE_TOKEN_MAX, size=input_len, dtype=np.int64)
    return tokens.tolist()


def generate_dataset(
    num_requests: int,
    input_len: int = 50_000,
    base_seed: int = 42,
) -> List[List[int]]:
    """Generate token IDs for multiple requests."""
    return [generate_request_tokens(input_len, seed=base_seed + i) for i in range(num_requests)]
