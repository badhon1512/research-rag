from __future__ import annotations

from hashlib import sha1
from math import sqrt
import re


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "how",
    "is",
    "it",
    "of",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "why",
}


class HashingEmbeddingModel:
    def __init__(self, dimensions: int = 2048) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            index = _stable_index(token, self.dimensions)
            vector[index] += 1.0
        return _normalize(vector)


class SentenceTransformerEmbeddingModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        local_files_only: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "SentenceTransformerEmbeddingModel needs the vector dependencies: "
                "pip install \"rag-explorer[vector]\""
            ) from error
        except ImportError as error:
            raise RuntimeError(
                f"SentenceTransformerEmbeddingModel could not import its installed dependencies: {error}"
            ) from error

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


def _stable_index(token: str, dimensions: int) -> int:
    digest = sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dimensions


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in TOKEN_RE.finditer(text))
        if token not in STOPWORDS
    ]


def _normalize(vector: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
