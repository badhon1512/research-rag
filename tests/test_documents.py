import tempfile
import unittest
from pathlib import Path

from rag_explorer.documents import (
    DocumentExtractionError,
    UnsupportedDocumentTypeError,
    load_document,
    load_documents,
    supported_extensions,
)


class DocumentLoadingTests(unittest.TestCase):
    def test_loads_plain_text_with_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "note.txt"
            path.write_text("A small RAG note.", encoding="utf-8")

            documents = load_document(path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["extension"], ".txt")
        self.assertEqual(documents[0].title, "Note")
        self.assertIn("RAG note", documents[0].text)

    def test_load_documents_skips_unsupported_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.md").write_text("# Note\n\nRAG text.", encoding="utf-8")
            (root / "image.png").write_bytes(b"not text")

            documents = load_documents(root)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["filename"], "note.md")

    def test_load_document_rejects_unknown_single_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"not text")

            with self.assertRaises(UnsupportedDocumentTypeError):
                load_document(path)

    def test_pdf_support_is_declared(self):
        self.assertIn(".pdf", supported_extensions())

    def test_pdf_without_dependency_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            try:
                import pypdf  # noqa: F401
            except ImportError:
                with self.assertRaisesRegex(DocumentExtractionError, "PDF support needs"):
                    load_document(path)


if __name__ == "__main__":
    unittest.main()
