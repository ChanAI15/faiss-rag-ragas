"""Ingest PDFs from the documents directory into the FAISS index."""

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR
from rag.ingestion.chunker import chunk_text
from rag.ingestion.extractor import extract_pdf_pages, extract_pdf_text
from rag.retrieval.vector_store import save_chunks


def extract_documents() -> list[tuple[str, str]]:
    """Extract PDF content only; this does not call Gemini or FAISS."""
    documents = [(path.name, extract_pdf_text(path)) for path in sorted(DOCUMENTS_DIR.glob("*.pdf"))]
    if not documents:
        raise FileNotFoundError(f"No readable PDFs found in {DOCUMENTS_DIR}")
    return documents


def extract_document_pages() -> list[tuple[str, list[dict[str, int | str]]]]:
    """Extract every PDF while retaining page boundaries for JSON export."""
    documents = [(path.name, extract_pdf_pages(path)) for path in sorted(DOCUMENTS_DIR.glob("*.pdf"))]
    if not documents:
        raise FileNotFoundError(f"No readable PDFs found in {DOCUMENTS_DIR}")
    return documents


def ingest_documents() -> int:
    chunks = [
        chunk
        for source, text in extract_documents()
        for chunk in chunk_text(text, source, CHUNK_SIZE, CHUNK_OVERLAP)
    ]
    save_chunks(chunks)
    return len(chunks)
