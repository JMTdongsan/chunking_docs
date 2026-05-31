from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class DenseTextEmbedder(Protocol):
    embedding_dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class DenseImageEmbedder(Protocol):
    embedding_dim: int

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        """Embed a batch of images."""


class HashingTextEmbedder:
    """Deterministic local fallback for tests and pipeline dry-runs."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = np.zeros(self.embedding_dim, dtype=np.float32)
            for token in text.split():
                index = hash(token) % self.embedding_dim
                vector[index] += 1.0
            norm = np.linalg.norm(vector)
            if norm:
                vector = vector / norm
            vectors.append(vector.tolist())
        return vectors
