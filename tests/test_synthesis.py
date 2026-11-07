"""Unit tests for Phase 2: Synthetic Dataset Generation, Deduplication, and Schemas."""

import pytest
from pydantic import ValidationError

from src.synthesis.prompt_templates import QuizQuestion, QuizResponse, SummaryResponse
from src.synthesis.deduplicator import DatasetDeduplicator
from src.synthesis.data_generator import DataGenerator
from src.synthesis.dataset_builder import DatasetBuilder


class TestPromptSchemas:
    def test_valid_quiz_question(self):
        q = QuizQuestion(
            question="What is the dominant differential voltage in CAN bus?",
            options={"A": "0V", "B": "1.0V", "C": "2.0V", "D": "2.5V"},
            correct_answer="C",
            explanation="Dominant state drives CAN_H to 3.5V and CAN_L to 1.5V (3.5 - 1.5 = 2.0V).",
            citation="Section 2: Physical Layer",
        )
        assert q.correct_answer == "C"
        assert len(q.options) == 4

    def test_invalid_correct_answer_rejected(self):
        with pytest.raises(ValidationError):
            QuizQuestion(
                question="Invalid question",
                options={"A": "1", "B": "2", "C": "3", "D": "4"},
                correct_answer="E",  # Invalid! Must be A, B, C, or D
                explanation="None",
                citation="None",
            )


class TestDeduplicator:
    def setup_method(self):
        self.dedup = DatasetDeduplicator(similarity_threshold=0.70, min_question_chars=20)

    def test_quality_filter_drops_short_or_refusal(self):
        bad_sample1 = {"instruction": "Q", "input": "C", "output": "Too short"}
        valid1, _ = self.dedup.is_valid_quality(bad_sample1)
        assert valid1 is False

        bad_sample2 = {"instruction": "Q", "input": "C", "output": "As an AI language model, I cannot answer."}
        valid2, _ = self.dedup.is_valid_quality(bad_sample2)
        assert valid2 is False

    def test_deduplication_catches_near_identical_outputs(self):
        out1 = "Câu hỏi: Trạng thái dominant của mạng CAN có hiệu điện thế chênh lệch là bao nhiêu?"
        out2 = "Câu hỏi: Trạng thái dominant của mạng CAN có mức hiệu điện thế chênh lệch là bao nhiêu?"

        sample1 = {"instruction": "Tạo câu hỏi", "input": "...", "output": out1}
        sample2 = {"instruction": "Tạo câu hỏi", "input": "...", "output": out2}

        assert self.dedup.process_sample(sample1) is True
        assert self.dedup.process_sample(sample2) is False  # Duplicate!


class TestDataGeneratorAndBuilder:
    def test_mock_generation_produces_all_task_types(self):
        gen = DataGenerator(use_mock=True)
        chunk = {
            "text": "In the dominant state, CAN_H goes to 3.5V and CAN_L drops to 1.5V, creating a 2V differential.",
            "source_file": "test.pdf",
            "page_number": 1,
            "chunk_id": 0,
        }

        for task in ["quiz", "summary", "troubleshooting"]:
            samples = gen.generate_for_chunk(chunk, task_type=task)
            assert len(samples) >= 1
            for sample in samples:
                assert "instruction" in sample
                assert "input" in sample
                assert "output" in sample
                assert len(sample["output"]) > 30

    def test_dataset_builder_split(self, tmp_path):
        gen = DataGenerator(use_mock=True)
        dedup = DatasetDeduplicator()
        builder = DatasetBuilder(generator=gen, deduplicator=dedup, val_split_ratio=0.25)

        # Create temporary dummy chunks.jsonl
        dummy_chunks_path = tmp_path / "dummy_chunks.jsonl"
        with open(dummy_chunks_path, "w", encoding="utf-8") as f:
            for i in range(8):
                f.write(f'{{"chunk_id": {i}, "text": "CAN controller node {i} maintains Transmit Error Counter (TEC) threshold {i*30}V."}}\n')

        out_dir = tmp_path / "synthetic_out"
        stats = builder.build_dataset(chunks_path=dummy_chunks_path, output_dir=out_dir)

        assert stats["total_chunks_processed"] == 8
        assert stats["train_count"] > 0
        assert stats["val_count"] > 0
        assert stats["train_count"] + stats["val_count"] == stats["total_accepted"]
        assert (out_dir / "train.jsonl").exists()
        assert (out_dir / "val.jsonl").exists()
