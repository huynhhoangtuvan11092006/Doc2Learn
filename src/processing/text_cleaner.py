"""Module for sanitizing, normalizing, and denoising raw extracted text from technical documents."""

import re
from typing import List, Optional


class TextCleaner:
    """Cleans and standardizes raw extracted text from PDFs using configurable regex rules."""

    def __init__(self, custom_noise_patterns: Optional[List[str]] = None):
        """Args:

        custom_noise_patterns: Optional list of regex patterns representing
        document-specific headers, footers, or disclaimers to strip out.
        """
        # Default technical document noise patterns (page numbers, common disclaimers)
        self.noise_patterns = [
            r"Page\s+\d+\s+of\s+\d+",                     # Page 1 of 50
            r"^\s*\d+\s*$",                                # Standalone page number on single line
            r"(?i)confidential\s+document",                # Confidential notice
            r"(?i)all\s+rights\s+reserved\.?",            # Copyright / All rights reserved
            r"(?i)draft\s+-\s+not\s+for\s+distribution",   # Draft notices
        ]
        if custom_noise_patterns:
            self.noise_patterns.extend(custom_noise_patterns)

    def clean(self, text: str) -> str:
        """Apply full cleaning pipeline on the input string.

        Steps:
        1. Remove noise patterns (headers/footers/disclaimers).
        2. Fix hyphenated word wrapping at line ends (e.g., 'trans- \nformer' -> 'transformer').
        3. Replace tabs and strange Unicode spaces with single standard spaces.
        4. Normalize excessive newlines (keep max 2 newlines for paragraph separation).
        5. Trim leading and trailing whitespace.
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = text

        # Step 1: Strip noise patterns
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)

        # Step 2: Fix hyphenated word breaks at line ends
        cleaned = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", cleaned)

        # Step 3: Replace non-breaking spaces & tabs with standard space
        cleaned = re.sub(r"[\t\r\f\v\u00A0]", " ", cleaned)

        # Step 4: Normalize spaces within lines (collapse multiple spaces into one)
        cleaned = re.sub(r"[ ]{2,}", " ", cleaned)

        # Step 5: Normalize consecutive newlines (max 2 consecutive newlines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()
