"""Multi-source content extraction module for Doc2Learn.

Extracts textual content from multiple formats:
1. Raw Text / Pasted Topics
2. Uploaded Documents (PDF, TXT, Markdown)
3. Web Page Articles (URLs)
4. YouTube Video Transcripts

Integrates seamlessly with TextCleaner and SemanticChunker from Phase 1.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional
import httpx
from bs4 import BeautifulSoup

from src.processing.chunker import SemanticChunker
from src.processing.text_cleaner import TextCleaner


# YouTube Video ID Extraction regex
YOUTUBE_REGEX = re.compile(
    r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
)


def extract_from_text(raw_text: str) -> str:
    """Validate and normalize user-pasted text."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Input text cannot be empty.")
    return raw_text.strip()


def extract_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from an uploaded document buffer (PDF, TXT, or Markdown).

    Args:
        file_bytes: Raw binary bytes of the file.
        filename: Name of the file with extension.

    Returns:
        Extracted text content.
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    ext = filename.lower().split(".")[-1] if "." in filename else ""

    if ext == "pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages.append(text)
            doc.close()
            combined = "\n\n".join(pages)
            if not combined.strip():
                raise ValueError("PDF does not contain any readable text.")
            return combined
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    elif ext in ("txt", "md", "markdown"):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="replace")
    else:
        raise ValueError(f"Unsupported file format: '.{ext}'. Supported: PDF, TXT, MD.")


def extract_from_web_url(url: str, timeout: float = 10.0) -> str:
    """Scrape and extract readable article text from a web URL.

    Args:
        url: Web URL string.
        timeout: Request timeout in seconds.

    Returns:
        Extracted article text without boilerplate/navigation HTML.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        raise ValueError(f"Failed to fetch content from URL '{url}': {str(e)}")

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        tag.decompose()

    # Priority 1: Check article / main tags
    article_tag = soup.find("article") or soup.find("main")
    if article_tag:
        paragraphs = [p.get_text().strip() for p in article_tag.find_all(["p", "h1", "h2", "h3", "li"])]
    else:
        # Fallback to all paragraph and header tags
        paragraphs = [p.get_text().strip() for p in soup.find_all(["p", "h1", "h2", "h3", "li"])]

    text = "\n\n".join(p for p in paragraphs if len(p.split()) >= 4)
    if not text.strip():
        # Fallback to general get_text
        text = soup.get_text(separator="\n", strip=True)

    if not text.strip():
        raise ValueError(f"Could not extract readable article text from '{url}'.")

    return text


def extract_from_youtube_url(url: str) -> str:
    """Extract transcript from a YouTube video URL using youtube-transcript-api.

    Args:
        url: Full YouTube link (watch, youtu.be, or shorts).

    Returns:
        Full combined transcript text.
    """
    match = YOUTUBE_REGEX.search(url)
    if not match:
        raise ValueError(f"Invalid YouTube URL format: '{url}'")

    video_id = match.group(1)

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        # Support fallback across multiple language codes
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "vi", "es", "de", "fr"]
        )
        full_text = " ".join(item.get("text", "") for item in transcript_list)
        if not full_text.strip():
            raise ValueError("Retrieved YouTube transcript is empty.")
        return full_text
    except Exception as e:
        raise ValueError(f"Failed to retrieve YouTube transcript for video '{video_id}': {str(e)}")


def process_and_chunk_content(
    raw_text: str,
    source_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Clean and segment any extracted text into standardized chunks (Phase 1 pipeline).

    Args:
        raw_text: Raw text from any source.
        source_metadata: Optional dictionary describing source (e.g. filename, url, type).
        chunk_size: Target token count per chunk (default 512).
        chunk_overlap: Overlapping tokens between chunks (default 50).

    Returns:
        List of chunk dictionaries ready for LLM generation.
    """
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(raw_text)

    chunker = SemanticChunker(max_tokens=chunk_size, overlap_tokens=chunk_overlap)
    meta = source_metadata or {"source_type": "text"}
    chunks = chunker.chunk_text(cleaned_text, metadata=meta)

    return chunks
