"""NLP Metrics computation module for Doc2Learn evaluation.

Implements ROUGE-L, BLEU-4, Perplexity, and Format Compliance Rate
for quantitative benchmarking of generated quizzes and technical summaries.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """Compute average ROUGE-L F1 score between predictions and references.

    Args:
        predictions: List of generated texts.
        references: List of reference (ground truth) texts.

    Returns:
        Average ROUGE-L F1 score (0.0 to 1.0).
    """
    if not predictions or not references or len(predictions) != len(references):
        return 0.0

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        return float(sum(scores) / len(scores))
    except ImportError:
        # Fallback LCS (Longest Common Subsequence) calculation if rouge_score not installed
        scores = []
        for pred, ref in zip(predictions, references):
            p_tokens = pred.strip().lower().split()
            r_tokens = ref.strip().lower().split()
            if not p_tokens or not r_tokens:
                scores.append(0.0)
                continue
            lcs_len = _lcs_length(p_tokens, r_tokens)
            prec = lcs_len / len(p_tokens) if p_tokens else 0.0
            rec = lcs_len / len(r_tokens) if r_tokens else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            scores.append(f1)
        return float(sum(scores) / len(scores)) if scores else 0.0


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Helper to compute Longest Common Subsequence length for fallback ROUGE-L."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def compute_bleu_4(predictions: List[str], references: List[str]) -> float:
    """Compute average BLEU-4 score (n-gram order 4) between predictions and references.

    Args:
        predictions: List of generated texts.
        references: List of reference texts.

    Returns:
        BLEU-4 score (0.0 to 1.0).
    """
    if not predictions or not references or len(predictions) != len(references):
        return 0.0

    try:
        import sacrebleu
        # sacrebleu expects references as list of lists: [[ref1_item1, ref2_item1...]]
        # Here we compute sentence-level or corpus-level score scaled to 0.0 - 1.0
        bleu = sacrebleu.corpus_bleu(predictions, [[r] for r in references])
        return float(bleu.score / 100.0)
    except ImportError:
        # Fallback simple n-gram BLEU-4 with brevity penalty
        scores = []
        for pred, ref in zip(predictions, references):
            scores.append(_simple_sentence_bleu(pred, ref, max_n=4))
        return float(sum(scores) / len(scores)) if scores else 0.0


def _simple_sentence_bleu(pred: str, ref: str, max_n: int = 4) -> float:
    """Fallback simple BLEU computation with geometric mean of n-grams."""
    p_tokens = pred.strip().lower().split()
    r_tokens = ref.strip().lower().split()

    if not p_tokens or not r_tokens:
        return 0.0

    # Brevity penalty
    c = len(p_tokens)
    r = len(r_tokens)
    bp = math.exp(1 - r / c) if c < r else 1.0

    precisions = []
    for n in range(1, max_n + 1):
        if len(p_tokens) < n:
            precisions.append(0.0)
            continue
        p_ngrams = [tuple(p_tokens[i:i + n]) for i in range(len(p_tokens) - n + 1)]
        r_ngrams = [tuple(r_tokens[i:i + n]) for i in range(len(r_tokens) - n + 1)]
        r_counts = {}
        for ng in r_ngrams:
            r_counts[ng] = r_counts.get(ng, 0) + 1

        matches = 0
        for ng in p_ngrams:
            if r_counts.get(ng, 0) > 0:
                matches += 1
                r_counts[ng] -= 1

        prec = matches / len(p_ngrams) if p_ngrams else 0.0
        precisions.append(prec)

    # If any precision is 0, smoothed BLEU
    smooth = 1e-9
    log_prec_sum = sum(math.log(p if p > 0 else smooth) for p in precisions)
    geo_mean = math.exp(log_prec_sum / max_n)
    return float(bp * geo_mean)


def extract_json_from_text(text: str) -> Optional[Any]:
    """Extract and parse JSON object or list from text (even if embedded in markdown blocks)."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding markdown ```json ... ``` blocks
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } or [ ... ]
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    return None


def validate_quiz_structure(item: Any) -> bool:
    """Validate whether an item conforms to standard Quiz JSON schema.

    Expected schema:
      - question: str (non-empty)
      - options: dict with keys A, B, C, D (or at least 3-4 options)
      - correct_answer: str (matches one of the option keys)
      - explanation: str (optional/recommended)
    """
    if not isinstance(item, dict):
        return False

    # Check question
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        return False

    # Check options
    options = item.get("options")
    if not isinstance(options, dict) or len(options) < 3:
        return False

    # Check correct answer
    correct = item.get("correct_answer")
    if not isinstance(correct, str) or not correct.strip():
        return False

    # Clean correct answer key (e.g. 'A.' -> 'A')
    clean_correct = correct.strip().rstrip(".")[0].upper() if correct.strip() else ""
    opt_keys = [str(k).strip().rstrip(".")[0].upper() for k in options.keys() if str(k).strip()]

    if clean_correct not in opt_keys and correct not in options:
        return False

    return True


def compute_format_compliance(
    predictions: List[str], expected_format: str = "quiz"
) -> Dict[str, Any]:
    """Compute the Format Compliance Rate for generated outputs.

    Args:
        predictions: List of raw model outputs.
        expected_format: 'quiz' for JSON MCQs or 'summary' for Markdown Slide/Summary.

    Returns:
        Dict with total, valid_count, compliance_rate (0.0 to 1.0), and details.
    """
    if not predictions:
        return {
            "total": 0,
            "valid_count": 0,
            "compliance_rate": 0.0,
            "format": expected_format,
        }

    valid_count = 0

    if expected_format.lower() == "quiz":
        for pred in predictions:
            parsed = extract_json_from_text(pred)
            if parsed is None:
                continue

            if isinstance(parsed, list):
                # If a list of quizzes is returned, all or at least 1 must be valid
                if parsed and all(validate_quiz_structure(q) for q in parsed):
                    valid_count += 1
            elif isinstance(parsed, dict):
                if validate_quiz_structure(parsed):
                    valid_count += 1
                elif "quizzes" in parsed and isinstance(parsed["quizzes"], list):
                    if parsed["quizzes"] and all(validate_quiz_structure(q) for q in parsed["quizzes"]):
                        valid_count += 1

    elif expected_format.lower() in ("summary", "slide"):
        for pred in predictions:
            text = pred.strip()
            # Must contain markdown headers (#, ##, or ###) or Slide keywords AND bullet points (- or *)
            has_headers = bool(re.search(r"(^|\n)#+\s+|\bSlide\s+\d+", text, re.IGNORECASE))
            has_bullets = bool(re.search(r"(^|\n)\s*[-*•]\s+", text))
            has_minimum_length = len(text.split()) >= 8

            if (has_headers or has_bullets) and has_minimum_length:
                valid_count += 1
    else:
        # Generic non-empty format check
        valid_count = sum(1 for p in predictions if len(p.strip()) > 10)

    total = len(predictions)
    rate = valid_count / total if total > 0 else 0.0

    return {
        "total": total,
        "valid_count": valid_count,
        "compliance_rate": round(rate, 4),
        "format": expected_format,
    }


def compute_perplexity(loss: float) -> float:
    """Compute Perplexity from Cross-Entropy Loss: PPL = exp(loss).

    Args:
        loss: Average cross-entropy loss.

    Returns:
        Perplexity value (capped at 10000.0 to prevent overflow).
    """
    try:
        ppl = math.exp(loss)
        return round(min(ppl, 10000.0), 4)
    except OverflowError:
        return 10000.0


def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    expected_format: str = "quiz",
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Comprehensive evaluation aggregator function.

    Computes ROUGE-L, BLEU-4, Format Compliance Rate, and Perplexity.
    """
    rouge_l = compute_rouge_l(predictions, references)
    bleu_4 = compute_bleu_4(predictions, references)
    compliance = compute_format_compliance(predictions, expected_format=expected_format)

    results: Dict[str, Any] = {
        "rouge_l": round(rouge_l, 4),
        "bleu_4": round(bleu_4, 4),
        "format_compliance_rate": compliance["compliance_rate"],
        "valid_formats": compliance["valid_count"],
        "total_samples": compliance["total"],
        "expected_format": expected_format,
    }

    if loss is not None:
        results["loss"] = round(loss, 4)
        results["perplexity"] = compute_perplexity(loss)

    return results

