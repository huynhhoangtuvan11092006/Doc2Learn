"""Unit tests for the Phase 1 Data Pipeline: TextCleaner and SemanticChunker."""

import pytest
from src.processing.text_cleaner import TextCleaner
from src.processing.chunker import SemanticChunker


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_strip_page_numbers_and_confidential(self):
        dirty = "CONFIDENTIAL DOCUMENT\nIntroduction to CAN Bus.\nPage 1 of 50\nEnd of section."
        cleaned = self.cleaner.clean(dirty)
        assert "Page 1 of 50" not in cleaned
        assert "CONFIDENTIAL DOCUMENT" not in cleaned
        assert "Introduction to CAN Bus." in cleaned

    def test_fix_hyphenated_line_breaks(self):
        dirty = "The microcon- \ntroller executes the firmware."
        cleaned = self.cleaner.clean(dirty)
        assert "microcontroller" in cleaned
        assert "microcon- \ntroller" not in cleaned

    def test_normalize_spaces_and_newlines(self):
        dirty = "Word1    Word2\n\n\n\n\nWord3"
        cleaned = self.cleaner.clean(dirty)
        assert cleaned == "Word1 Word2\n\nWord3"


class TestSemanticChunker:
    def setup_method(self):
        self.chunker = SemanticChunker(max_tokens=50, overlap_tokens=10)

    def test_chunking_token_limits(self):
        long_text = (
            "The Controller Area Network (CAN) is a robust vehicle bus standard designed to allow "
            "microcontrollers and devices to communicate with each other's applications without a host computer. "
            "It is a message-based protocol, designed originally for multiplex electrical wiring within automobiles "
            "to save on copper, but it is also used in many other contexts."
        )
        chunks = self.chunker.chunk_text(long_text, metadata={"source": "test"})
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["token_count"] <= 60  # Allow slight tolerance for sentence boundary
            assert "chunk_id" in chunk
            assert chunk["source"] == "test"

    def test_empty_text_returns_empty_list(self):
        assert self.chunker.chunk_text("") == []
        assert self.chunker.chunk_text("   ") == []
