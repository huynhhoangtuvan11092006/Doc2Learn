"""Inference module for comparing Base Model vs Fine-tuned LoRA Model.

Supports both GPU-based real model generation (using Hugging Face + PEFT)
and local offline mock evaluation (--mock) for fast development and pipeline verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_evaluation_data(data_path: str | Path) -> List[Dict[str, Any]]:
    """Load evaluation samples from a JSONL file."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {path}")

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def generate_mock_base_output(instruction: str, context: str) -> str:
    """Simulate a raw Base LLM output (verbose prose, markdown conversational filler, imperfect JSON)."""
    # Base models typically return chatty conversational preamble and non-strict formatting
    return (
        f"Sure! Here is some information about the topic based on your instruction:\n\n"
        f"Regarding the document: {context[:120]}...\n"
        f"In CAN bus systems, errors can occur when voltage reaches 12V or 999 TEC.\n\n"
        f"Question: What is a dominant bit in CAN?\n"
        f"1. A bit with voltage\n"
        f"2. A recessive bit\n"
        f"3. CAN-High at 3.5V\n"
        f"The answer is probably option 3 because CAN-High drives differential voltage."
    )


def generate_mock_finetuned_output(instruction: str, context: str) -> str:
    """Simulate a Fine-tuned LoRA output (strict JSON structure, grounded technical facts)."""
    # Fine-tuned model strictly follows the Alpaca JSON schema trained during Phase 3
    sample_quiz = {
        "question": "What are the voltage levels of CAN-High and CAN-Low during a dominant bit transmission?",
        "options": {
            "A": "CAN-High is 2.5V, CAN-Low is 2.5V (ΔV = 0.0V)",
            "B": "CAN-High is 3.5V, CAN-Low is 1.5V (ΔV = 2.0V)",
            "C": "CAN-High is 5.0V, CAN-Low is 0.0V (ΔV = 5.0V)",
            "D": "CAN-High is 1.5V, CAN-Low is 3.5V (ΔV = -2.0V)",
        },
        "correct_answer": "B",
        "explanation": "During a dominant state (logic 0), CAN-High is actively driven to 3.5V and CAN-Low to 1.5V, creating a 2.0V differential voltage."
    }
    return json.dumps(sample_quiz, ensure_ascii=False, indent=2)


def run_gpu_inference(
    model_name_or_path: str,
    adapter_path: Optional[str],
    prompts: List[str],
    max_new_tokens: int = 512,
) -> List[str]:
    """Execute real GPU inference using Hugging Face Transformers and PEFT."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[*] Loading base model: {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"[*] Attaching LoRA adapter from: {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    outputs = []

    print(f"[*] Generating predictions for {len(prompts)} prompts...")
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Decode only the generated portion
            input_len = inputs["input_ids"].shape[1]
            response_text = tokenizer.decode(
                generated_ids[0][input_len:], skip_special_tokens=True
            ).strip()
            outputs.append(response_text)

    return outputs


def run_inference_benchmark(
    val_data_path: str | Path = "data/synthetic/val.jsonl",
    output_dir: str | Path = "data/evaluation",
    mock: bool = True,
    base_model_name: str = "unsloth/llama-3-8b-Instruct",
    adapter_path: str = "models/lora-adapter",
) -> Dict[str, Path]:
    """Run inference for both Base Model and Fine-tuned Model and save predictions.

    Args:
        val_data_path: Path to validation dataset.
        output_dir: Directory to save generated output files.
        mock: If True, uses offline simulation without downloading weights.
        base_model_name: Hugging Face model identifier for base model.
        adapter_path: Directory containing trained LoRA adapter safetensors.

    Returns:
        Dict mapping model type to saved JSONL file path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_evaluation_data(val_data_path)
    print(f"[*] Loaded {len(samples)} evaluation samples from {val_data_path}")

    base_results = []
    finetuned_results = []

    if mock:
        print("[*] Running in MOCK / DRY-RUN mode (local CPU fast simulation)...")
        for idx, sample in enumerate(samples):
            instr = sample.get("instruction", "")
            inp = sample.get("input", "")
            ref = sample.get("output", "")

            # Generate mock outputs
            base_pred = generate_mock_base_output(instr, inp)
            ft_pred = generate_mock_finetuned_output(instr, inp)

            base_results.append({
                "id": idx,
                "instruction": instr,
                "input": inp,
                "reference": ref,
                "prediction": base_pred,
                "model": "base_model",
            })

            finetuned_results.append({
                "id": idx,
                "instruction": instr,
                "input": inp,
                "reference": ref,
                "prediction": ft_pred,
                "model": "finetuned_model",
            })
    else:
        print("[*] Running in REAL GPU mode...")
        # Prepare Llama-3 format prompts
        prompts = [
            f"<|start_header_id|>user<|end_header_id|>\n\n{s.get('instruction', '')}\n\n{s.get('input', '')}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            for s in samples
        ]

        print("[*] 1/2: Generating Base Model outputs...")
        raw_base_outputs = run_gpu_inference(base_model_name, None, prompts)

        print("[*] 2/2: Generating Fine-tuned LoRA outputs...")
        raw_ft_outputs = run_gpu_inference(base_model_name, adapter_path, prompts)

        for idx, (s, b_out, ft_out) in enumerate(zip(samples, raw_base_outputs, raw_ft_outputs)):
            base_results.append({
                "id": idx,
                "instruction": s.get("instruction", ""),
                "input": s.get("input", ""),
                "reference": s.get("output", ""),
                "prediction": b_out,
                "model": "base_model",
            })
            finetuned_results.append({
                "id": idx,
                "instruction": s.get("instruction", ""),
                "input": s.get("input", ""),
                "reference": s.get("output", ""),
                "prediction": ft_out,
                "model": "finetuned_model",
            })

    # Save to disk
    base_file = out_dir / "output_base.jsonl"
    ft_file = out_dir / "output_finetuned.jsonl"

    with open(base_file, "w", encoding="utf-8") as f:
        for item in base_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(ft_file, "w", encoding="utf-8") as f:
        for item in finetuned_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[✓] Base model outputs saved to: {base_file}")
    print(f"[✓] Fine-tuned model outputs saved to: {ft_file}")

    return {
        "base_output_path": base_file,
        "finetuned_output_path": ft_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation inference benchmark")
    parser.add_argument("--val-data", default="data/synthetic/val.jsonl", help="Validation dataset path")
    parser.add_argument("--output-dir", default="data/evaluation", help="Directory to save outputs")
    parser.add_argument("--mock", action="store_true", default=False, help="Run mock offline inference")
    parser.add_argument("--base-model", default="unsloth/llama-3-8b-Instruct", help="Base model identifier")
    parser.add_argument("--adapter-path", default="models/lora-adapter", help="LoRA adapter directory")

    args = parser.parse_args()
    run_inference_benchmark(
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        mock=args.mock,
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
    )


if __name__ == "__main__":
    main()

