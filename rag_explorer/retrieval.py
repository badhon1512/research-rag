from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
import re

from rag_explorer.chroma_store import ChromaVectorStore
from rag_explorer.chunking import Chunk
from rag_explorer.embeddings import HashingEmbeddingModel, SentenceTransformerEmbeddingModel
from rag_explorer.vector_store import InMemoryVectorStore


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("TfidfRetriever needs at least one chunk")

        self.chunks = chunks
        self.term_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())

        total = len(chunks)
        self.idf = {
            token: log((1 + total) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }
        self.vectors = [self._vectorize_counts(counts) for counts in self.term_counts]

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_counts = Counter(tokenize(query))
        query_vector = self._vectorize_counts(query_counts)

        results = [
            RetrievalResult(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        ranked = sorted(results, key=lambda result: result.score, reverse=True)
        return [result for result in ranked[:top_k] if result.score > 0]

    def _vectorize_counts(self, counts: Counter[str]) -> dict[str, float]:
        return {
            token: count * self.idf.get(token, 0.0)
            for token, count in counts.items()
        }


class VectorRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedding_model: HashingEmbeddingModel | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("VectorRetriever needs at least one chunk")

        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self.vector_store = InMemoryVectorStore()
        for chunk in chunks:
            self.vector_store.add(chunk, self.embedding_model.embed(chunk.text))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_vector = self.embedding_model.embed(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        return [
            RetrievalResult(chunk=result.chunk, score=result.score)
            for result in results
        ]


class ChromaIndexer:
    def __init__(
        self,
        store_dir: str | Path = "data/chroma",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        local_files_only: bool = False,
    ) -> None:
        self.embedding_model = SentenceTransformerEmbeddingModel(
            model_name=model_name,
            local_files_only=local_files_only,
        )
        self.store_dir = store_dir

    def index(self, chunks: list[Chunk], reset: bool = True) -> int:
        store = ChromaVectorStore(store_dir=self.store_dir, reset=reset)
        vectors = [self.embedding_model.embed(chunk.text) for chunk in chunks]
        store.add(chunks, vectors)
        return store.count()


class ChromaRetriever:
    def __init__(
        self,
        store_dir: str | Path = "data/chroma",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        local_files_only: bool = True,
    ) -> None:
        self.embedding_model = SentenceTransformerEmbeddingModel(
            model_name=model_name,
            local_files_only=local_files_only,
        )
        self.vector_store = ChromaVectorStore(store_dir=store_dir)

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_vector = self.embedding_model.embed(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        return [
            RetrievalResult(chunk=result.chunk, score=result.score)
            for result in results
        ]


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
