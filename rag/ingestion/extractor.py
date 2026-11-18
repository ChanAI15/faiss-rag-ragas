"""PDF text extraction with PyMuPDF."""

from pathlib import Path

import fitz


def extract_pdf_text(path: Path) -> str:
    """Extract text from every page of a PDF in page order."""
    with fitz.open(path) as document:
        return "\n".join(page.get_text() for page in document)


def extract_pdf_pages(pdf_path: Path | str) -> list[dict[str, int | str]]:
    """Extract text from a PDF and retain its original page boundaries."""
    pages: list[dict[str, int | str]] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text()
            pages.append({
                "page_number": page_number,
                "text": text,
                "source": Path(pdf_path).name,
            })
    return pages
