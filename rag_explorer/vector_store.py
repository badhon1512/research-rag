from __future__ import annotations

from dataclasses import dataclass

from rag_explorer.chunking import Chunk


@dataclass(frozen=True)
class VectorRecord:
    chunk: Chunk
    vector: list[float]


@dataclass(frozen=True)
class VectorSearchResult:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.records: list[VectorRecord] = []

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        self.records.append(VectorRecord(chunk=chunk, vector=vector))

    def search(self, query_vector: list[float], top_k: int = 3) -> list[VectorSearchResult]:
        results = [
            VectorSearchResult(chunk=record.chunk, score=dot_product(query_vector, record.vector))
            for record in self.records
        ]
        ranked = sorted(results, key=lambda result: result.score, reverse=True)
        return [result for result in ranked[:top_k] if result.score > 0]


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))
