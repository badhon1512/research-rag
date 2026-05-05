from __future__ import annotations

from dataclasses import dataclass
import re

from rag_explorer.retrieval import RetrievalResult, tokenize


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Citation:
    source: str
    title: str
    chunk_id: str
    score: float


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    citations: list[Citation]


class ExtractiveAnswerer:
    def answer(self, question: str, results: list[RetrievalResult]) -> Answer:
        if not results:
            return Answer(
                question=question,
                answer="I could not find relevant evidence in the indexed documents.",
                citations=[],
            )

        query_terms = set(tokenize(question))
        selected: list[str] = []

        for result in results:
            clean_text = _remove_markdown_headings(result.chunk.text)
            sentences = SENTENCE_RE.split(clean_text.replace("\n", " "))
            best_sentence = max(
                sentences,
                key=lambda sentence: len(query_terms.intersection(tokenize(sentence))),
                default=clean_text,
            ).strip()
            if best_sentence and best_sentence not in selected:
                selected.append(best_sentence)

        citations = [
            Citation(
                source=result.chunk.source,
                title=result.chunk.title,
                chunk_id=result.chunk.id,
                score=round(result.score, 4),
            )
            for result in results
        ]
        return Answer(question=question, answer=" ".join(selected), citations=citations)


def _remove_markdown_headings(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    return "\n".join(lines).strip()
