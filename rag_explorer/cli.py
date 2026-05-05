from __future__ import annotations

import argparse

from rag_explorer.chunking import chunk_documents
from rag_explorer.documents import load_documents
from rag_explorer.pipeline import RagPipeline
from rag_explorer.retrieval import ChromaIndexer, ChromaRetriever, TfidfRetriever, VectorRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over the local RAG corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--data-dir", default="data/raw")
    ask_parser.add_argument("--top-k", type=int, default=3)

    chunks_parser = subparsers.add_parser("chunks", help="Inspect document chunks")
    chunks_parser.add_argument("--data-dir", default="data/raw")
    chunks_parser.add_argument("--max-chars", type=int, default=900)
    chunks_parser.add_argument("--overlap", type=int, default=120)
    chunks_parser.add_argument("--preview-chars", type=int, default=160)

    index_parser = subparsers.add_parser("index", help="Build a persistent Chroma vector index")
    index_parser.add_argument("--data-dir", default="data/raw")
    index_parser.add_argument("--store-dir", default="data/chroma")
    index_parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    index_parser.add_argument("--max-chars", type=int, default=900)
    index_parser.add_argument("--overlap", type=int, default=120)
    index_parser.add_argument("--no-reset", action="store_true")
    index_parser.add_argument("--local-files-only", action="store_true")

    retrieve_parser = subparsers.add_parser("retrieve", help="Inspect retrieved chunks for a question")
    retrieve_parser.add_argument("question")
    retrieve_parser.add_argument("--data-dir", default="data/raw")
    retrieve_parser.add_argument("--store-dir", default="data/chroma")
    retrieve_parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    retrieve_parser.add_argument("--top-k", type=int, default=3)
    retrieve_parser.add_argument("--max-chars", type=int, default=900)
    retrieve_parser.add_argument("--overlap", type=int, default=120)
    retrieve_parser.add_argument("--preview-chars", type=int, default=260)
    retrieve_parser.add_argument("--retriever", choices=["tfidf", "vector", "chroma"], default="tfidf")

    args = parser.parse_args()

    if args.command == "ask":
        pipeline = RagPipeline(data_dir=args.data_dir, top_k=args.top_k)
        response = pipeline.ask(args.question)
        print(response.answer)
        if response.citations:
            print("\nCitations:")
            for citation in response.citations:
                print(
                    f"- {citation.title} | {citation.source} | "
                    f"chunk={citation.chunk_id} | score={citation.score}"
                )
    elif args.command == "chunks":
        documents = load_documents(args.data_dir)
        chunks = chunk_documents(
            documents,
            max_chars=args.max_chars,
            overlap_chars=args.overlap,
        )
        print(f"Loaded {len(documents)} document(s). Created {len(chunks)} chunk(s).\n")
        for index, chunk in enumerate(chunks, start=1):
            preview = " ".join(chunk.text.split())
            if len(preview) > args.preview_chars:
                preview = f"{preview[: args.preview_chars].rstrip()}..."
            print(f"{index}. {chunk.title}")
            print(f"   source={chunk.source}")
            print(f"   chunk={chunk.id} chars={chunk.start}-{chunk.end}")
            print(f"   preview={preview}\n")
    elif args.command == "index":
        documents = load_documents(args.data_dir)
        chunks = chunk_documents(
            documents,
            max_chars=args.max_chars,
            overlap_chars=args.overlap,
        )
        indexer = ChromaIndexer(
            store_dir=args.store_dir,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
        )
        count = indexer.index(chunks, reset=not args.no_reset)
        print(f"Loaded {len(documents)} document(s). Indexed {len(chunks)} chunk(s).")
        print(f"Chroma store: {args.store_dir}")
        print(f"Collection count: {count}")
    elif args.command == "retrieve":
        if args.retriever == "tfidf":
            documents = load_documents(args.data_dir)
            chunks = chunk_documents(
                documents,
                max_chars=args.max_chars,
                overlap_chars=args.overlap,
            )
            retriever = TfidfRetriever(chunks)
            searched_count = len(chunks)
        else:
            if args.retriever == "vector":
                documents = load_documents(args.data_dir)
                chunks = chunk_documents(
                    documents,
                    max_chars=args.max_chars,
                    overlap_chars=args.overlap,
                )
                retriever = VectorRetriever(chunks)
                searched_count = len(chunks)
            else:
                documents = []
                retriever = ChromaRetriever(store_dir=args.store_dir, model_name=args.model_name)
                searched_count = retriever.vector_store.count()
        results = retriever.search(args.question, top_k=args.top_k)

        print(f"Question: {args.question}")
        print(f"Retriever: {args.retriever}")
        print(f"Loaded {len(documents)} document(s). Searched {searched_count} chunk(s).")
        print(f"Returned {len(results)} result(s).\n")

        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            preview = " ".join(chunk.text.split())
            if len(preview) > args.preview_chars:
                preview = f"{preview[: args.preview_chars].rstrip()}..."
            print(f"{index}. score={result.score:.4f} title={chunk.title}")
            print(f"   source={chunk.source}")
            print(f"   chunk={chunk.id} chars={chunk.start}-{chunk.end}")
            print(f"   preview={preview}\n")


if __name__ == "__main__":
    main()
