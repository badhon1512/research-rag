from rag_explorer.chunking import chunk_document
from rag_explorer.documents import Document
import unittest


class ChunkingTests(unittest.TestCase):
    def test_chunk_document_preserves_source_metadata(self):
        document = Document(
            source="example.md",
            title="Example",
            text="# Example\n\nRAG retrieves evidence before generation.",
            metadata={"kind": "sample"},
        )

        chunks = chunk_document(document, max_chars=80, overlap_chars=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source, "example.md")
        self.assertEqual(chunks[0].metadata["kind"], "sample")
        self.assertIn("retrieves evidence", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
