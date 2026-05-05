from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_explorer.chunking import Chunk
from rag_explorer.vector_store import VectorSearchResult


class ChromaVectorStore:
    def __init__(
        self,
        store_dir: str | Path = "data/chroma",
        collection_name: str = "rag_explorer_chunks",
        reset: bool = False,
    ) -> None:
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError(
                "ChromaVectorStore needs the vector dependencies: "
                "pip install \"rag-explorer[vector]\""
            ) from error

        self.client = chromadb.PersistentClient(path=str(store_dir))
        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        self.collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=vectors,
            metadatas=[_metadata_from_chunk(chunk) for chunk in chunks],
        )

    def count(self) -> int:
        return int(self.collection.count())

    def search(self, query_vector: list[float], top_k: int = 3) -> list[VectorSearchResult]:
        response = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[VectorSearchResult] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            chunk = _chunk_from_record(chunk_id, text, metadata or {})
            results.append(VectorSearchResult(chunk=chunk, score=_score_from_distance(distance)))
        return results


def _metadata_from_chunk(chunk: Chunk) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "source": chunk.source,
        "title": chunk.title,
        "start": chunk.start,
        "end": chunk.end,
    }
    for key, value in chunk.metadata.items():
        metadata[f"metadata.{key}"] = value
    return metadata


def _chunk_from_record(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    chunk_metadata = {
        key.removeprefix("metadata."): str(value)
        for key, value in metadata.items()
        if key.startswith("metadata.")
    }
    return Chunk(
        id=chunk_id,
        text=text,
        source=str(metadata.get("source", "")),
        title=str(metadata.get("title", "")),
        start=int(metadata.get("start", 0)),
        end=int(metadata.get("end", len(text))),
        metadata=chunk_metadata,
    )


def _score_from_distance(distance: float) -> float:
    return 1.0 - float(distance)
