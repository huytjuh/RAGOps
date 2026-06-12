from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class BertEmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    fallback_dimensions: int = 32


class BertEmbeddingService:
    """Create BERT sentence embeddings, with a deterministic local fallback."""

    def __init__(self, config: BertEmbeddingConfig | None = None) -> None:
        self.config = config or BertEmbeddingConfig()
        self._model = None

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        model = self._load_sentence_transformer()
        if model is not None:
            vectors = model.encode(texts, normalize_embeddings=True)
            return [list(map(float, vector)) for vector in vectors]
        return [self._hash_embedding(text) for text in texts]

    def _load_sentence_transformer(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None

        self._model = SentenceTransformer(self.config.model_name)
        return self._model

    def _hash_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for index in range(self.config.fallback_dimensions):
            byte = digest[index % len(digest)]
            values.append((byte / 127.5) - 1.0)
        return values
