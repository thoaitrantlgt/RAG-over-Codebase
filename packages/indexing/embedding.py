import hashlib
import math
from pathlib import Path
from typing import Protocol

from .text import tokenize


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...


class HashingEmbeddingProvider:
    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class FastEmbedEmbeddingProvider:
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        cache_dir: str | None = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed is required for real local embeddings. "
                "Install it with: pip install fastembed"
            ) from exc

        self.model_name = model_name
        self.cache_dir = str(Path(cache_dir).resolve()) if cache_dir else None
        self._model = TextEmbedding(model_name=model_name, cache_dir=self.cache_dir)
        self.dimensions = fastembed_dimensions(TextEmbedding, model_name)

    def embed(self, text: str) -> list[float]:
        vector = next(iter(self._model.embed([text])))
        return [float(value) for value in vector]


def create_embedding_provider(
    *,
    provider: str = "hashing",
    dimensions: int = 128,
    model_name: str = "jinaai/jina-embeddings-v2-base-code",
    cache_dir: str | None = None,
) -> EmbeddingProvider:
    if provider == "hashing":
        return HashingEmbeddingProvider(dimensions=dimensions)
    if provider == "fastembed":
        return FastEmbedEmbeddingProvider(model_name=model_name, cache_dir=cache_dir)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def fastembed_dimensions(text_embedding_cls, model_name: str) -> int:
    for model in text_embedding_cls.list_supported_models():
        if model.get("model") == model_name:
            return int(model["dim"])
    probe = text_embedding_cls(model_name=model_name)
    vector = next(iter(probe.embed(["dimension probe"])))
    return len(vector)


def embedding_text(chunk: dict) -> str:
    return "\n".join(
        [
            chunk.get("symbol_name", ""),
            chunk.get("summary", ""),
            chunk.get("code_body", ""),
        ]
    )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
