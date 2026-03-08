"""
Shared fixtures and helpers used across all test modules.

dummy_embed produces deterministic, unit-length vectors so tests are
fast (no fastembed install needed) and fully reproducible.
"""
import hashlib
import math

import pytest

from semantic_cache.backends.memory import MemoryBackend


# --------------------------------------------------------------------------- #
#  Deterministic fake embedder (dim=16, no external deps)                     #
# --------------------------------------------------------------------------- #

def dummy_embed(text: str) -> list[float]:
    """
    Embed text as a deterministic unit-length vector in R^16.
    Same text → same vector. Different text → different vector.
    """
    seed_bytes = hashlib.sha256(text.encode()).digest()
    raw = [
        int.from_bytes(seed_bytes[i : i + 2], "big") / 65535.0
        for i in range(0, 32, 2)  # 16 values
    ]
    magnitude = math.sqrt(sum(x * x for x in raw))
    return [x / magnitude for x in raw]


def near_embed(text: str, noise: float = 0.03) -> list[float]:
    """
    Return a vector close to dummy_embed(text) — simulates a semantic paraphrase.
    The noise is small enough that cosine similarity stays above 0.85.
    """
    import random
    rng = random.Random(text)
    base = dummy_embed(text)
    perturbed = [x + rng.gauss(0, noise) for x in base]
    magnitude = math.sqrt(sum(x * x for x in perturbed))
    return [x / magnitude for x in perturbed]


# --------------------------------------------------------------------------- #
#  Fixtures                                                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture
def backend():
    b = MemoryBackend()
    yield b
    b.clear_sync()


@pytest.fixture
def cache(backend):
    from semantic_cache import SemanticCache
    return SemanticCache(backend=backend, embed_fn=dummy_embed, threshold=0.85)
