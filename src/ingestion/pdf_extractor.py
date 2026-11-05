"""Module for extracting text and metadata from PDF technical documents using PyMuPDF (fitz)."""

from pathlib import Path
from typing import Dict, List, Any
import fitz  # PyMuPDF


class PDFExtractor:
    """High-performance PDF text extractor powered by PyMuPDF (MuPDF C-engine)."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {self.file_path}")

    def extract_pages(self) -> List[Dict[str, Any]]:
        """Extract text and metadata from each page of the PDF document.

        Returns:
            List of dictionaries containing:
                - 'page_number': 1-indexed page number
                - 'raw_text': Extracted raw string from the page
                - 'char_count': Character count of raw text
                - 'metadata': General document metadata (title, author, etc.)
        """
        extracted_pages = []
        doc = fitz.open(self.file_path)
        doc_meta = doc.metadata or {}

        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text("text")

                extracted_pages.append(
                    {
                        "page_number": page_index + 1,
                        "raw_text": text,
                        "char_count": len(text),
                        "file_name": self.file_path.name,
                        "doc_title": doc_meta.get("title", ""),
                    }
                )
        finally:
            doc.close()

        return extracted_pages

    def extract_full_text(self) -> str:
        """Convenience method to extract all text concatenated from the entire document."""
        pages = self.extract_pages()
        return "\n\n".join(page["raw_text"] for page in pages if page["raw_text"].strip())
