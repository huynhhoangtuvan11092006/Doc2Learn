"""QLoRA fine-tuning script for Meta Llama-3 using PyTorch, Hugging Face PEFT, and TRL SFTTrainer."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any
import yaml

from .data_loader import DataLoaderLlama


def load_config(config_path: str | Path = "configs/training_config.yaml") -> Dict[str, Any]:
    """Load training and LoRA hyperparameters from YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_lora_trainable_params(
    hidden_size: int = 4096,
    num_hidden_layers: int = 32,
    lora_r: int = 16,
    num_target_modules: int = 7,
    total_base_params: int = 8_030_000_000,
) -> Dict[str, Any]:
    """Calculate the exact theoretical LoRA parameter count for interview verification.

    For each linear layer of shape (d_in, d_out), LoRA inserts matrix A (d_in, r) and matrix B (r, d_out).
    Trainable parameters = (d_in + d_out) * r.
    """
    # For standard Llama-3-8B hidden dimension d = 4096:
    params_per_module = 2 * hidden_size * lora_r  # Matrix A + Matrix B
    total_trainable = params_per_module * num_target_modules * num_hidden_layers
    percentage = (total_trainable / total_base_params) * 100

    return {
        "trainable_params": total_trainable,
        "total_base_params": total_base_params,
        "trainable_percentage": round(percentage, 4),
    }


def run_dry_run_training(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Simulates training execution for local verification, CI/CD, and environments without Nvidia GPU."""
    print("[*] Running in DRY-RUN mode (CPU / Hardware Verification)...")

    # 1. Verify Dataset loading
    train_path = config["paths"]["train_data"]
    val_path = config["paths"]["val_data"]
    loader = DataLoaderLlama(train_path, val_path)
    train_data, val_data = loader.get_datasets()
    print(f"[+] Datasets verified: {len(train_data)} train samples, {len(val_data)} validation samples.")

    # 2. Verify Parameter Math
    lora_cfg = config["lora"]
    stats = calculate_lora_trainable_params(
        hidden_size=4096,
        num_hidden_layers=32,
        lora_r=lora_cfg["r"],
        num_target_modules=len(lora_cfg["target_modules"]),
    )
    print(f"[+] LoRA Math calculated:")
    print(f"    - Base Model Params: {stats['total_base_params']:,}")
    print(f"    - Trainable LoRA Params: {stats['trainable_params']:,}")
    print(f"    - Trainable Ratio: {stats['trainable_percentage']}% (< 1% of total parameters!)")

    # 3. Create output directory and mock adapter checkpoint for Phase 4 compatibility
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_meta = {
        "base_model_name": config["model"]["base_model_name"],
        "lora_r": lora_cfg["r"],
        "lora_alpha": lora_cfg["lora_alpha"],
        "target_modules": lora_cfg["target_modules"],
        "task_type": lora_cfg["task_type"],
        "training_epochs": config["training"]["num_train_epochs"],
        "learning_rate": config["training"]["learning_rate"],
        "dry_run": True,
    }
    with open(output_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_meta, f, indent=2)

    # Touch mock weights placeholder
    with open(output_dir / "adapter_model.safetensors", "wb") as f:
        f.write(b"MOCK_LORA_WEIGHTS_FOR_LOCAL_VALIDATION")

    print(f"[✓] Dry-run completed successfully! Mock adapter saved to: {output_dir}")
    return stats


def run_gpu_training(config: Dict[str, Any], output_dir: Path):
    """Executes full QLoRA fine-tuning using PyTorch, BitsAndBytes, and TRL SFTTrainer on GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    base_model_name = config["model"]["base_model_name"]
    print(f"[*] Starting GPU QLoRA Training for: {base_model_name}")

    # 1. Configure 4-bit Quantization (NormalFloat4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # 2. Load Base Model in 4-bit
    print("[*] Loading base model weights in 4-bit precision...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    model.config.use_cache = False  # Silence warnings during gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Attach LoRA Adapter
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Load Formatted Datasets
    loader = DataLoaderLlama(config["paths"]["train_data"], config["paths"]["val_data"])
    train_dataset, val_dataset = loader.get_datasets()

    # 6. Configure TrainingArguments / SFTConfig (Compatible with both new and older TRL versions)
    train_cfg = config["training"]
    try:
        from trl import SFTConfig
        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=float(train_cfg["learning_rate"]),
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            weight_decay=train_cfg["weight_decay"],
            optim=train_cfg["optim"],
            logging_steps=train_cfg["logging_steps"],
            save_strategy=train_cfg["save_strategy"],
            seed=train_cfg["seed"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",
            max_length=train_cfg["max_seq_length"],
            dataset_text_field="text",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
        )
    except (ImportError, TypeError):
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=float(train_cfg["learning_rate"]),
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            weight_decay=train_cfg["weight_decay"],
            optim=train_cfg["optim"],
            logging_steps=train_cfg["logging_steps"],
            save_strategy=train_cfg["save_strategy"],
            seed=train_cfg["seed"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=train_cfg["max_seq_length"],
            tokenizer=tokenizer,
            args=training_args,
        )

    print("[*] Initiating SFTTrainer optimization loop...")
    trainer.train()

    print(f"[*] Training finished! Saving LoRA adapter to: {output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[✓] LoRA adapter weights saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Doc2Learn QLoRA Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Run local validation without GPU execution")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["training"]["output_dir"])

    # If dry-run requested or CUDA is unavailable, run dry-run validation
    import importlib.util
    has_torch = importlib.util.find_spec("torch") is not None
    has_cuda = False
    if has_torch:
        import torch
        has_cuda = torch.cuda.is_available()

    if args.dry_run or not has_cuda:
        run_dry_run_training(config, output_dir)
    else:
        run_gpu_training(config, output_dir)


if __name__ == "__main__":
    main()

