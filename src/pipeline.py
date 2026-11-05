"""End-to-end data pipeline script: PDF -> Cleaned Text -> Semantic Chunks -> JSONL dataset."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

from src.ingestion.pdf_extractor import PDFExtractor
from src.processing.text_cleaner import TextCleaner
from src.processing.chunker import SemanticChunker


def run_data_pipeline(
    pdf_path: str | Path,
    output_path: str | Path = "data/processed/chunks.jsonl",
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> List[Dict[str, Any]]:
    """Executes the extraction, sanitization, and chunking pipeline.

    Args:
        pdf_path: Path to the target technical PDF.
        output_path: Destination JSONL file path for processed chunks.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Overlap tokens between consecutive chunks.

    Returns:
        List of processed chunk dictionaries.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting extraction for: {pdf_path}")
    extractor = PDFExtractor(pdf_path)
    pages = extractor.extract_pages()
    print(f"[+] Extracted {len(pages)} pages successfully.")

    cleaner = TextCleaner()
    chunker = SemanticChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    all_chunks: List[Dict[str, Any]] = []
    total_chunk_counter = 0

    print("[*] Sanitizing and chunking content...")
    for page in pages:
        raw_text = page["raw_text"]
        cleaned_text = cleaner.clean(raw_text)

        if not cleaned_text:
            continue

        page_meta = {
            "source_file": pdf_path.name,
            "page_number": page["page_number"],
            "doc_title": page.get("doc_title", ""),
        }

        page_chunks = chunker.chunk_text(cleaned_text, metadata=page_meta)
        for chunk in page_chunks:
            chunk["global_chunk_id"] = total_chunk_counter
            total_chunk_counter += 1
            all_chunks.append(chunk)

    # Save to JSONL
    print(f"[*] Saving {len(all_chunks)} chunks to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[✓] Pipeline complete! Output written to {output_path}")
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doc2Learn Data Preprocessing Pipeline")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--output", type=str, default="data/processed/chunks.jsonl", help="Output JSONL path")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap tokens per chunk")

    args = parser.parse_args()
    run_data_pipeline(args.pdf, args.output, args.max_tokens, args.overlap)
