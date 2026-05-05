from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

from rag_explorer.documents import Document


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    title: str
    start: int
    end: int
    metadata: dict[str, str]


def chunk_documents(
    documents: list[Document],
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars, overlap_chars=overlap_chars))
    return chunks


def chunk_document(document: Document, max_chars: int = 900, overlap_chars: int = 120) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")

    paragraphs = [p.strip() for p in document.text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    current_start = 0
    cursor = 0

    for paragraph in paragraphs:
        start = document.text.find(paragraph, cursor)
        if start == -1:
            start = cursor
        cursor = start + len(paragraph)

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            if not current:
                current_start = start
            current = candidate
            continue

        if current:
            chunks.append(_make_chunk(document, current, current_start))

        if len(paragraph) <= max_chars:
            current = paragraph
            current_start = start
        else:
            chunks.extend(_split_long_text(document, paragraph, start, max_chars, overlap_chars))
            current = ""
            current_start = cursor

    if current:
        chunks.append(_make_chunk(document, current, current_start))

    return chunks


def _split_long_text(
    document: Document,
    text: str,
    absolute_start: int,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max_chars - overlap_chars
    for relative_start in range(0, len(text), step):
        piece = text[relative_start : relative_start + max_chars].strip()
        if piece:
            chunks.append(_make_chunk(document, piece, absolute_start + relative_start))
        if relative_start + max_chars >= len(text):
            break
    return chunks


def _make_chunk(document: Document, text: str, start: int) -> Chunk:
    digest = sha1(f"{document.source}:{start}:{text}".encode("utf-8")).hexdigest()[:12]
    return Chunk(
        id=digest,
        text=text,
        source=document.source,
        title=document.title,
        start=start,
        end=start + len(text),
        metadata=document.metadata,
    )
