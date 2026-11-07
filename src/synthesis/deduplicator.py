"""Deduplication and quality filtering engine for synthetic instruction-following datasets."""

import re
from typing import Dict, List, Set, Any, Tuple


class DatasetDeduplicator:
    """Filters duplicate samples using token-level Jaccard similarity and enforces pedagogical quality checks."""

    def __init__(self, similarity_threshold: float = 0.75, min_question_chars: int = 25):
        """Args:

        similarity_threshold: Jaccard similarity limit above which a sample is
        flagged as duplicate (default: 0.75). min_question_chars: Minimum
        acceptable character length for questions.
        """
        self.similarity_threshold = similarity_threshold
        self.min_question_chars = min_question_chars
        self.seen_signatures: List[Tuple[Set[str], str]] = []
        self.stats = {
            "total_evaluated": 0,
            "passed": 0,
            "dropped_duplicate": 0,
            "dropped_malformed": 0,
        }

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Convert text into normalized word token set (lowercased, punctuation-stripped)."""
        clean = re.sub(r"[^\w\s]", "", text.lower())
        tokens = {t for t in clean.split() if len(t) > 2}
        return tokens

    def _compute_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard index: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def is_valid_quality(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify that the Alpaca sample satisfies essential schema and pedagogical standards."""
        instruction = sample.get("instruction", "").strip()
        context = sample.get("input", "").strip()
        output = sample.get("output", "").strip()

        if not instruction or not output:
            return False, "Missing instruction or output"

        if len(output) < self.min_question_chars:
            return False, f"Output too short (< {self.min_question_chars} chars)"

        # Check for placeholder / generic failure strings
        lower_output = output.lower()
        if "as an ai language model" in lower_output or "i cannot answer" in lower_output:
            return False, "Contains AI refusal boilerplate"

        return True, "OK"

    def is_duplicate(self, key_text: str) -> bool:
        """Check if `key_text` is semantically redundant compared to previously accepted samples."""
        tokens = self._tokenize(key_text)
        if not tokens:
            return False

        for existing_tokens, _ in self.seen_signatures:
            similarity = self._compute_jaccard(tokens, existing_tokens)
            if similarity >= self.similarity_threshold:
                return True

        self.seen_signatures.append((tokens, key_text))
        return False

    def process_sample(self, sample: Dict[str, Any], deduplication_key: str = None) -> bool:
        """Process a single sample through quality check and deduplication.

        Returns True if the sample is accepted, False if dropped.
        """
        self.stats["total_evaluated"] += 1

        # Step 1: Quality Check
        valid, _ = self.is_valid_quality(sample)
        if not valid:
            self.stats["dropped_malformed"] += 1
            return False

        # Step 2: Deduplication Check
        key = deduplication_key or sample.get("output", "")
        if self.is_duplicate(key):
            self.stats["dropped_duplicate"] += 1
            return False

        self.stats["passed"] += 1
        return True

    def filter_dataset(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a list of Alpaca samples in-place, returning only unique, high-quality records."""
        accepted = []
        for sample in samples:
            if self.process_sample(sample):
                accepted.append(sample)
        return accepted

    def get_summary(self) -> Dict[str, Any]:
        """Return cumulative processing metrics."""
        return dict(self.stats)
