"""Unit tests for Phase 3: QLoRA Training Configuration, Llama-3 Chat Template, and Data Loader."""

import json
from pathlib import Path
import pytest

from src.training.data_loader import DataLoaderLlama, format_llama3_chat
from src.training.train import load_config, calculate_lora_trainable_params, run_dry_run_training


class TestTrainingConfig:
    def test_training_config_loads_correctly(self):
        config = load_config("configs/training_config.yaml")

        assert "model" in config
        assert "quantization" in config
        assert "lora" in config
        assert "training" in config
        assert "paths" in config

        # Verify key hyperparameters
        assert config["quantization"]["load_in_4bit"] is True
        assert config["quantization"]["bnb_4bit_quant_type"] == "nf4"
        assert config["lora"]["r"] == 16
        assert config["lora"]["lora_alpha"] == 32
        assert len(config["lora"]["target_modules"]) == 7
        assert float(config["training"]["learning_rate"]) == 0.0002


class TestLlama3ChatFormatter:
    def test_special_tokens_integrity(self):
        instruction = "Biên soạn 1 câu hỏi trắc nghiệm về mạng CAN."
        context = "CAN High operates at 3.5V in dominant state."
        output = "Câu 1: Điện áp CAN High là bao nhiêu? A. 3.5V ..."

        formatted = format_llama3_chat(instruction, context, output)

        # Check required Meta Llama-3 special control tokens
        assert "<|begin_of_text|>" in formatted
        assert "<|start_header_id|>system<|end_header_id|>" in formatted
        assert "<|start_header_id|>user<|end_header_id|>" in formatted
        assert "<|start_header_id|>assistant<|end_header_id|>" in formatted
        assert formatted.count("<|eot_id|>") == 3  # System turn, User turn, Assistant turn

        # Verify content placement
        assert instruction in formatted
        assert context in formatted
        assert output in formatted


class TestDataLoaderAndMath:
    def test_lora_parameter_math(self):
        stats = calculate_lora_trainable_params(
            hidden_size=4096,
            num_hidden_layers=32,
            lora_r=16,
            num_target_modules=7,
            total_base_params=8_030_000_000,
        )
        assert stats["trainable_params"] == 29_360_128
        assert stats["trainable_percentage"] < 1.0  # Must be fraction of 1%
        assert round(stats["trainable_percentage"], 2) == 0.37

    def test_data_loader_formatting(self, tmp_path):
        sample_file = tmp_path / "temp_train.jsonl"
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "instruction": "Tạo câu hỏi",
                "input": "CAN_H = 3.5V",
                "output": "Đáp án: A",
            }) + "\n")

        loader = DataLoaderLlama(sample_file, sample_file)
        raw_records = loader.load_jsonl(sample_file)
        assert len(raw_records) == 1

        formatted_records = loader.format_records(raw_records)
        assert len(formatted_records) == 1
        assert "text" in formatted_records[0]
        assert "<|begin_of_text|>" in formatted_records[0]["text"]

    def test_dry_run_execution(self, tmp_path):
        config = load_config("configs/training_config.yaml")
        out_dir = tmp_path / "test_lora_out"

        stats = run_dry_run_training(config, out_dir)
        assert stats["trainable_params"] > 0
        assert (out_dir / "adapter_config.json").exists()
        assert (out_dir / "adapter_model.safetensors").exists()

