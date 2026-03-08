import asyncio
import logging
from typing import Callable

log = logging.getLogger(__name__)

# Type aliases used throughout the library
EmbedFn = Callable[[str], list[float]]


class FastEmbedder:
    """
    Default local embedder powered by fastembed (ONNX, no API key needed).

    The underlying model is held as a class-level singleton so multiple
    SemanticCache instances in the same process share one loaded model.
    """

    _model = None  # shared across all instances

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        if FastEmbedder._model is None:
            try:
                from fastembed import TextEmbedding
                FastEmbedder._model = TextEmbedding(model_name)
                log.debug("FastEmbedder: loaded model %s", model_name)
            except ImportError:
                raise ImportError(
                    "fastembed is required for the default embedder.\n"
                    "Install it with:  pip install semantic-cache[default]"
                )

    def embed(self, text: str) -> list[float]:
        embeddings = list(FastEmbedder._model.embed([text]))
        return embeddings[0].tolist()

    async def aembed(self, text: str) -> list[float]:
        """Run embed() in a thread executor so it doesn't block the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, text)
