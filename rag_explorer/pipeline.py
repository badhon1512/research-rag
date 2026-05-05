from __future__ import annotations

from pathlib import Path

from rag_explorer.chunking import chunk_documents
from rag_explorer.documents import load_documents
from rag_explorer.generation import Answer, ExtractiveAnswerer
from rag_explorer.retrieval import TfidfRetriever


class RagPipeline:
    def __init__(self, data_dir: str | Path = "data/raw", top_k: int = 3) -> None:
        documents = load_documents(data_dir)
        chunks = chunk_documents(documents)
        self.retriever = TfidfRetriever(chunks)
        self.answerer = ExtractiveAnswerer()
        self.top_k = top_k

    def ask(self, question: str) -> Answer:
        results = self.retriever.search(question, top_k=self.top_k)
        return self.answerer.answer(question, results)



if __name__ == "__main__":
    pipeline = RagPipeline()
    while True:
        user_input = input("\nEnter your question (or 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        answer = pipeline.ask(user_input)
        print(f"\nAnswer: {answer.answer}")
        if answer.citations:
            print("\nCitations:")
            for citation in answer.citations:
                print(f"- {citation.title} (source: {citation.source}, score: {citation.score})")