"""Unit tests for the evaluation module (metrics, hallucination, inference, and reports)."""

import json
from pathlib import Path
import pytest

from src.evaluation.metrics import (
    compute_rouge_l,
    compute_bleu_4,
    compute_format_compliance,
    compute_perplexity,
    evaluate_predictions,
    extract_json_from_text,
    validate_quiz_structure,
)
from src.evaluation.hallucination_checker import (
    extract_technical_entities,
    check_hallucination,
    batch_check_hallucination,
)
from src.evaluation.human_eval import run_human_evaluation
from src.evaluation.report_generator import (
    evaluate_model_outputs,
    generate_markdown_report,
    run_benchmark_pipeline,
)


class TestMetrics:
    """Test NLP scoring functions and format validators."""

    def test_rouge_l_identical(self):
        text = ["the CAN bus physical layer uses differential signaling"]
        score = compute_rouge_l(text, text)
        assert score == pytest.approx(1.0, rel=1e-2)

    def test_rouge_l_empty_or_mismatch(self):
        assert compute_rouge_l([], []) == 0.0
        assert compute_rouge_l(["apple"], ["banana"]) == 0.0
        assert compute_rouge_l(["a"], ["b", "c"]) == 0.0

    def test_bleu_4_identical(self):
        text = ["the CAN bus protocol defines frame format and arbitration"]
        score = compute_bleu_4(text, text)
        assert score > 0.80

    def test_perplexity_calculation(self):
        # exp(0) = 1.0
        assert compute_perplexity(0.0) == 1.0
        # exp(2.507) ≈ 12.268
        ppl = compute_perplexity(2.507)
        assert 12.0 < ppl < 13.0

    def test_extract_json_from_markdown(self):
        raw = 'Some text\n```json\n{"question": "What?", "options": {"A": "1", "B": "2", "C": "3", "D": "4"}, "correct_answer": "A"}\n```\nMore text'
        parsed = extract_json_from_text(raw)
        assert parsed is not None
        assert parsed["question"] == "What?"

    def test_format_compliance_quiz(self):
        valid_quiz = json.dumps({
            "question": "What is CAN?",
            "options": {"A": "Bus", "B": "Car", "C": "Bike", "D": "Plane"},
            "correct_answer": "A",
            "explanation": "Controller Area Network",
        })
        invalid_quiz = "This is just conversational text with no schema."

        res = compute_format_compliance([valid_quiz, invalid_quiz], expected_format="quiz")
        assert res["total"] == 2
        assert res["valid_count"] == 1
        assert res["compliance_rate"] == 0.5

    def test_format_compliance_summary(self):
        valid_summary = "## Slide 1: Physical Layer\n- Line 1: CAN-High at 3.5V\n- Line 2: CAN-Low at 1.5V"
        invalid_summary = "too short"

        res = compute_format_compliance([valid_summary, invalid_summary], expected_format="summary")
        assert res["total"] == 2
        assert res["valid_count"] == 1
        assert res["compliance_rate"] == 0.5

    def test_evaluate_predictions_aggregate(self):
        preds = ["hello world can bus protocol"]
        refs = ["hello world can bus protocol"]
        res = evaluate_predictions(preds, refs, expected_format="summary", loss=2.5)
        assert "rouge_l" in res
        assert "bleu_4" in res
        assert "format_compliance_rate" in res
        assert "perplexity" in res


class TestHallucinationChecker:
    """Test technical entity extraction and hallucination detection."""

    def test_extract_technical_entities(self):
        text = "Operating voltage is 2.5V, with dominant state reaching 3.5V and speed of 1 Mbps. CRC is 0x7FF."
        entities = extract_technical_entities(text)
        assert "2.5v" in entities
        assert "3.5v" in entities
        assert "1mbps" in entities
        assert "CRC" in entities
        assert "0x7ff" in entities

    def test_check_hallucination_accurate(self):
        ctx = "CAN-High is 3.5V and CAN-Low is 1.5V. Maximum bit rate is 1 Mbps."
        gen = "Dominant bit sets CAN-High to 3.5V and CAN-Low to 1.5V."
        res = check_hallucination(gen, ctx)
        assert res["hallucination_rate"] == 0.0
        assert res["is_hallucinated"] is False

    def test_check_hallucination_fake(self):
        ctx = "CAN-High is 3.5V and CAN-Low is 1.5V."
        gen = "At high speed, voltage spikes to 12V and error counter exceeds 999 TEC."
        res = check_hallucination(gen, ctx)
        assert res["hallucination_rate"] > 0.0
        assert res["is_hallucinated"] is True

    def test_batch_check_hallucination(self):
        ctxs = ["Standard 11-bit identifier.", "Dominant bit 2.0V differential."]
        gens = ["Uses 11-bit identifier.", "Uses 99-bit identifier and 50V."]
        res = batch_check_hallucination(gens, ctxs)
        assert res["total_samples"] == 2
        assert res["hallucinated_samples_count"] == 1
        assert res["clean_samples_count"] == 1


class TestHumanEvalAndReporting:
    """Test human-in-the-loop scoring and benchmark report generation."""

    def test_human_eval_automated(self, tmp_path):
        in_file = tmp_path / "preds.jsonl"
        with open(in_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": 0, "prediction": "test", "input": "ctx", "model": "test_m"}) + "\n")

        out_file = tmp_path / "human_scores.json"
        summary = run_human_evaluation(
            input_file=in_file,
            output_file=out_file,
            interactive=False,
            default_scores={"accuracy": 4.0, "format_quality": 4.5, "pedagogical_value": 4.1},
        )
        assert out_file.exists()
        assert summary["total_evaluated"] == 1
        assert summary["average_overall_score"] == 4.2

    def test_benchmark_pipeline_end_to_end(self, tmp_path):
        base_file = tmp_path / "base.jsonl"
        ft_file = tmp_path / "ft.jsonl"

        with open(base_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": 0,
                "input": "CAN idle is 2.5V.",
                "reference": "2.5V",
                "prediction": "Sure! It is 12V and 2.5V",
            }) + "\n")

        valid_quiz = json.dumps({
            "question": "What is CAN idle voltage?",
            "options": {"A": "2.5V", "B": "1.5V", "C": "3.5V", "D": "0V"},
            "correct_answer": "A",
            "explanation": "Idle is 2.5V",
        })
        with open(ft_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": 0,
                "input": "CAN idle is 2.5V.",
                "reference": valid_quiz,
                "prediction": valid_quiz,
            }) + "\n")

        md_report = tmp_path / "report.md"
        summary_json = tmp_path / "summary.json"

        res = run_benchmark_pipeline(
            base_file=base_file,
            finetuned_file=ft_file,
            human_eval_file=None,
            output_report_md=md_report,
            output_summary_json=summary_json,
        )

        assert md_report.exists()
        assert summary_json.exists()
        assert "base_model" in res
        assert "finetuned_model" in res
        assert res["finetuned_model"]["format_compliance_rate"] == 1.0
        assert res["base_model"]["format_compliance_rate"] == 0.0

