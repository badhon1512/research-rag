from rag_explorer.chunking import Chunk
from rag_explorer.retrieval import TfidfRetriever, VectorRetriever
import unittest


class RetrievalTests(unittest.TestCase):
    def test_retriever_ranks_relevant_chunk_first(self):
        chunks = [
            Chunk("a", "Chunking splits documents into passages.", "a.md", "Chunking", 0, 42, {}),
            Chunk("b", "Evaluation checks answer support and citations.", "b.md", "Eval", 0, 46, {}),
        ]

        results = TfidfRetriever(chunks).search("How do citations support evaluation?", top_k=2)

        self.assertEqual(results[0].chunk.id, "b")

    def test_vector_retriever_returns_matching_chunk(self):
        chunks = [
            Chunk("a", "Chunking splits documents into passages.", "a.md", "Chunking", 0, 42, {}),
            Chunk("b", "Evaluation checks answer support and citations.", "b.md", "Eval", 0, 46, {}),
        ]

        results = VectorRetriever(chunks).search("evaluation citations", top_k=2)

        self.assertEqual(results[0].chunk.id, "b")


if __name__ == "__main__":
    unittest.main()
