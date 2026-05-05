from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class UnsupportedDocumentTypeError(ValueError):
    pass


class DocumentExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    title: str
    metadata: dict[str, str]


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def supported_extensions() -> set[str]:
    return set(SUPPORTED_EXTENSIONS)


def load_documents(data_dir: str | Path) -> list[Document]:
    root_dir = Path(data_dir)
    if not root_dir.exists():
        return []

    documents: list[Document] = []
    for file_path in sorted(root_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        documents.extend(load_document(file_path))
    return documents


def load_document(path: str | Path) -> list[Document]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return []

    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentTypeError(f"Unsupported document type: {extension}")

    if extension in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8")
    else:
        text = extract_text_from_pdf(file_path)

    return [
        Document(
            source=str(file_path),
            text=text,
            title=_title_from_text(text, file_path),
            metadata={"extension": extension, "filename": file_path.name},
        )
    ]


def extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise DocumentExtractionError(
            "PDF support needs the optional dependency: pip install \"rag-explorer[pdf]\""
        ) from error

    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise DocumentExtractionError(f"Failed to extract text from PDF {path}: {error}") from error


def _title_from_text(text: str, path: Path) -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("#"):
            return clean.lstrip("#").strip() or _title_from_path(path)
    return _title_from_path(path)


def _title_from_path(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").title()
