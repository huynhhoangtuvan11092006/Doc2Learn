"""Human-in-the-loop evaluation module for Doc2Learn.

Allows subject matter experts / reviewers to score generated quizzes and summaries
across 3 key criteria:
1. Technical Accuracy (1-5)
2. Format / Schema Quality (1-5)
3. Educational / Pedagogical Value (1-5)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def evaluate_sample_interactively(
    sample: Dict[str, Any], sample_idx: int, total_samples: int
) -> Dict[str, Any]:
    """Prompt human reviewer for ratings via CLI."""
    print(f"\n{'='*70}")
    print(f" Sample {sample_idx + 1} / {total_samples}")
    print(f"{'='*70}")
    print(f"[Model]: {sample.get('model', 'unknown')}")
    print(f"\n--- [Source Context] ---\n{sample.get('input', '')}")
    print(f"\n--- [Generated Output] ---\n{sample.get('prediction', '')}")
    print(f"{'-'*70}")

    def prompt_score(prompt_text: str) -> float:
        while True:
            try:
                val = input(f"{prompt_text} (1-5): ").strip()
                score = float(val)
                if 1.0 <= score <= 5.0:
                    return score
                print("  [!] Please enter a number between 1.0 and 5.0.")
            except ValueError:
                print("  [!] Invalid number format.")

    acc = prompt_score("1. Technical Accuracy")
    fmt = prompt_score("2. Format Quality")
    ped = prompt_score("3. Pedagogical Value")
    notes = input("Optional Notes (Enter to skip): ").strip()

    overall = round((acc + fmt + ped) / 3.0, 2)
    return {
        "id": sample.get("id", sample_idx),
        "model": sample.get("model", "unknown"),
        "accuracy": acc,
        "format_quality": fmt,
        "pedagogical_value": ped,
        "overall_score": overall,
        "notes": notes,
    }


def run_human_evaluation(
    input_file: str | Path,
    output_file: str | Path = "data/evaluation/human_eval_results.json",
    interactive: bool = True,
    default_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run human evaluation process over a JSONL output file.

    Args:
        input_file: Path to model predictions JSONL.
        output_file: Path to save evaluation scores JSON.
        interactive: If True, asks user in terminal. If False, uses simulated/default scores.
        default_scores: Pre-defined scores for non-interactive / automated testing.
    """
    in_path = Path(input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    samples = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    if interactive:
        print(f"[*] Starting Human Evaluation for {len(samples)} samples...")
        for i, s in enumerate(samples):
            score_data = evaluate_sample_interactively(s, i, len(samples))
            results.append(score_data)
    else:
        # Automated / simulated human scoring for CI/CD test pipelines
        defaults = default_scores or {
            "accuracy": 4.5,
            "format_quality": 4.8,
            "pedagogical_value": 4.2,
        }
        overall = round(
            (defaults["accuracy"] + defaults["format_quality"] + defaults["pedagogical_value"]) / 3.0,
            2,
        )
        for i, s in enumerate(samples):
            results.append({
                "id": s.get("id", i),
                "model": s.get("model", "unknown"),
                "accuracy": defaults["accuracy"],
                "format_quality": defaults["format_quality"],
                "pedagogical_value": defaults["pedagogical_value"],
                "overall_score": overall,
                "notes": "Automated evaluation baseline",
            })

    # Compute aggregate scores
    n = len(results)
    avg_overall = round(sum(r["overall_score"] for r in results) / n, 2) if n > 0 else 0.0
    summary = {
        "total_evaluated": n,
        "average_overall_score": avg_overall,
        "evaluations": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[✓] Human evaluation saved to: {out_path} (Average Score: {avg_overall}/5.0)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run human-in-the-loop scoring")
    parser.add_argument("--input-file", default="data/evaluation/output_finetuned.jsonl", help="Predictions file")
    parser.add_argument("--output-file", default="data/evaluation/human_eval_results.json", help="Output summary file")
    parser.add_argument("--auto", action="store_true", default=False, help="Run non-interactively with default scores")

    args = parser.parse_args()
    run_human_evaluation(
        input_file=args.input_file,
        output_file=args.output_file,
        interactive=not args.auto,
    )


if __name__ == "__main__":
    main()

