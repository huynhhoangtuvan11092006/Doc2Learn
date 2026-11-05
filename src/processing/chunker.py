"""Module for splitting continuous technical documents into semantically coherent chunks with overlap."""

from typing import Dict, List, Any
import tiktoken


class SemanticChunker:
    """Splits cleaned text into overlapping token-bounded chunks respecting paragraph boundaries."""

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        encoding_name: str = "cl100k_base",
    ):
        """Args:

        max_tokens: Maximum tokens allowed per chunk (default: 512, suitable for
        LLM context window). overlap_tokens: Number of tokens overlapping
        between consecutive chunks (default: 64). encoding_name: Tiktoken
        encoding scheme (default: cl100k_base).
        """
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be strictly less than max_tokens.")

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to cl100k_base if specified encoding fails
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Returns the number of tokens in the given text."""
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into semantic chunks with specified token limit and overlap.

        Splitting strategy:
        1. Split initially by double newlines (paragraphs).
        2. Progressively assemble paragraphs until reaching `max_tokens`.
        3. Slide forward while maintaining `overlap_tokens` from the tail of the previous chunk.
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []

        current_paragraphs = []
        current_token_count = 0
        chunk_index = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If a single paragraph is larger than max_tokens, split it by sentences
            if para_tokens > self.max_tokens:
                sentences = [s.strip() for s in para.split(". ") if s.strip()]
                for sentence in sentences:
                    sentence_str = sentence + ". "
                    s_tokens = self.count_tokens(sentence_str)
                    if current_token_count + s_tokens > self.max_tokens and current_paragraphs:
                        chunk_text = " ".join(current_paragraphs).strip()
                        chunks.append(self._create_chunk_record(chunk_index, chunk_text, metadata))
                        chunk_index += 1
                        # Retain overlap
                        current_paragraphs, current_token_count = self._compute_overlap(current_paragraphs)

                    current_paragraphs.append(sentence_str)
                    current_token_count += s_tokens
            else:
                if current_token_count + para_tokens > self.max_tokens and current_paragraphs:
                    chunk_text = "\n\n".join(current_paragraphs).strip()
                    chunks.append(self._create_chunk_record(chunk_index, chunk_text, metadata))
                    chunk_index += 1
                    # Retain overlap
                    current_paragraphs, current_token_count = self._compute_overlap(current_paragraphs)

                current_paragraphs.append(para)
                current_token_count += para_tokens

        # Flush any remaining text in the buffer
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs).strip()
            chunks.append(self._create_chunk_record(chunk_index, chunk_text, metadata))

        return chunks

    def _compute_overlap(self, paragraphs: List[str]) -> tuple[List[str], int]:
        """Keep paragraphs from the tail only if their tokens fit within `overlap_tokens`."""
        retained = []
        token_sum = 0
        for para in reversed(paragraphs):
            t = self.count_tokens(para)
            if token_sum + t <= self.overlap_tokens:
                retained.insert(0, para)
                token_sum += t
            else:
                break
        return retained, token_sum

    def _create_chunk_record(self, index: int, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Package chunk string with enriched metadata."""
        return {
            "chunk_id": index,
            "text": text,
            "token_count": self.count_tokens(text),
            "char_count": len(text),
            **metadata,
        }
