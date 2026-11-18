"""CLI entry point for ingesting PDFs and asking questions."""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS + Gemini RAG demo")
    commands = parser.add_subparsers(dest="command", required=True)
    extract_parser = commands.add_parser("extract", help="View PDF text without indexing or generation")
    extract_parser.add_argument("--full", action="store_true", help="Print all extracted text instead of a preview")
    chunk_parser = commands.add_parser("chunk", help="Chunk extraction JSON without embedding or indexing")
    chunk_parser.add_argument("--chunk-size", type=int, default=500, help="Words per chunk (default: 500)")
    chunk_parser.add_argument("--overlap", type=int, default=80, help="Overlapping words (default: 80)")
    commands.add_parser("embed", help="Embed chunk JSON and save vectors without building FAISS")
    commands.add_parser("index", help="Build FAISS from saved embedding JSON")
    commands.add_parser("ingest", help="Extract PDFs and build the FAISS index")
    search_parser = commands.add_parser("search", help="Search chunks by similarity without generation")
    search_parser.add_argument("question")
    search_parser.add_argument("--top-k", type=int, default=4)
    retrieve_parser = commands.add_parser("retrieve", help="Retrieve formatted context and sources without generation")
    retrieve_parser.add_argument("question")
    retrieve_parser.add_argument("--top-k", type=int, default=4)
    query_parser = commands.add_parser("query", help="Ask a question about indexed documents")
    query_parser.add_argument("question")
    query_parser.add_argument("--top-k", type=int, default=4)
    eval_parser = commands.add_parser("evaluate", help="Run RAGAS evaluation on the golden test dataset")
    eval_parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation questions to run")
    eval_parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks per question (default: 4)")
    eval_parser.add_argument("--judge-model", type=str, default=None, help="Judge LLM model name (default: from config)")
    eval_parser.add_argument("--reuse-samples", type=str, default=None, help="Path to cached samples JSON to re-evaluate")
    args = parser.parse_args()

    if args.command == "extract":
        from rag.ingestion.extractor import extract_pdf_pages

        pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No readable PDFs found in {DOCUMENTS_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for pdf_path in pdfs:
            source = pdf_path.name
            pages = extract_pdf_pages(pdf_path)
            output_path = OUTPUT_DIR / f"{source.rsplit('.', 1)[0]}.json"
            payload = {"source": source, "page_count": len(pages), "pages": pages}
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            text = "\n".join(str(page["text"]) for page in pages)
            print(f"Saved {len(pages)} pages to {output_path.relative_to(OUTPUT_DIR.parent)}")
            print(f"\n--- {source} ({len(text)} extracted characters) ---\n")
            if args.full:
                print(text)
            else:
                preview_length = 1_000
                print(text[:preview_length])
                if len(text) > preview_length:
                    print("\n[Preview truncated. Use --full to print the whole document.]")
        return
    if args.command == "ingest":
        from rag.ingestion.pipeline import ingest_documents

        print(f"Indexed {ingest_documents()} chunks.")
        return
    if args.command == "chunk":
        from rag.ingestion.chunk_pipeline import chunk_extraction_file

        extraction_files = sorted(path for path in OUTPUT_DIR.glob("*.json") if not path.name.endswith(".chunks.json"))
        if not extraction_files:
            raise FileNotFoundError("No extraction JSON found. Run `python main.py extract` first.")
        for path in extraction_files:
            output_path = chunk_extraction_file(path, args.chunk_size, args.overlap)
            print(f"Saved chunks to {output_path.relative_to(PROJECT_ROOT)}")
        return
    if args.command == "embed":
        from rag.ingestion.embedding_pipeline import embed_chunk_file

        chunk_files = sorted(OUTPUT_DIR.glob("*.chunks.json"))
        if not chunk_files:
            raise FileNotFoundError("No chunk JSON found. Run `python main.py chunk` first.")
        for path in chunk_files:
            output_path = embed_chunk_file(path)
            print(f"Saved embeddings to {output_path.relative_to(PROJECT_ROOT)}")
        return
    if args.command == "index":
        from rag.retrieval.index_pipeline import index_embedding_file

        embedding_files = sorted(OUTPUT_DIR.glob("*.embeddings.json"))
        if not embedding_files:
            raise FileNotFoundError("No embedding JSON found. Run `python main.py embed` first.")
        for path in embedding_files:
            count = index_embedding_file(path)
            print(f"Built FAISS index with {count} vectors from {path.relative_to(PROJECT_ROOT)}")
        return
    if args.command == "search":
        from rag.retrieval.search import search_documents

        results = search_documents(args.question, top_k=args.top_k)
        for result in results:
            print("\n--- Retrieved Chunk ---")
            print("Vector ID:", result["vector_id"])
            print("Similarity:", round(result["score"], 4))
            print("Source:", result["source"])
            print("Pages:", result["page_numbers"])
            print("Text:", result["text"][:300])
        return
    if args.command == "retrieve":
        from rag.retrieval.retriever import retrieve_context

        result = retrieve_context(args.question, top_k=args.top_k)
        print("Question:", result["question"])
        print("Retrieved chunks:", len(result["chunks"]))
        print("Sources:", result["sources"])
        print("\nContext:\n", result["context"][:2000])
        return
    if args.command == "evaluate":
        from rag.config import EVAL_JUDGE_MODEL
        from rag.evaluation.runner import evaluate_rag

        judge = args.judge_model or EVAL_JUDGE_MODEL
        evaluate_rag(
            limit=args.limit,
            top_k=args.top_k,
            judge_model=judge,
            reuse_samples_path=args.reuse_samples,
        )
        return
    from rag.generation.generator import generate_answer

    result = generate_answer(args.question, top_k=args.top_k)

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved Chunks (sent to the model):")
    for source in result["sources"]:
        print(
            f"- Chunk {source['chunk_id']}: {source['document']}, "
            f"pages {source['pages']}"
        )


if __name__ == "__main__":
    main()
