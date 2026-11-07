"""Data loader and formatter converting Alpaca-format JSONL into Meta Llama-3 Chat Template datasets."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert technical curriculum engineer and assessment designer for engineering domains. "
    "Your responses must be grounded strictly in the provided technical context with exact factual accuracy."
)

LLAMA3_CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{user_message}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{assistant_message}<|eot_id|>"
)


def format_llama3_chat(
    instruction: str,
    context: str,
    output: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Format an Alpaca record into Meta Llama-3 instruction chat format.

    Args:
        instruction: Task instruction.
        context: Technical context text (from input chunk).
        output: Expected ground-truth response.
        system_prompt: Optional system persona prompt.

    Returns:
        Formatted string containing Llama-3 special control tokens.
    """
    sys_p = system_prompt or DEFAULT_SYSTEM_PROMPT
    if context.strip():
        user_msg = f"{instruction.strip()}\n\n[TÀI LIỆU THAM KHẢO]:\n{context.strip()}"
    else:
        user_msg = instruction.strip()

    formatted = LLAMA3_CHAT_TEMPLATE.format(
        system_prompt=sys_p.strip(),
        user_message=user_msg,
        assistant_message=output.strip(),
    )
    return formatted


class DataLoaderLlama:
    """Loads synthetic JSONL datasets and prepares formatted records for SFTTrainer."""

    def __init__(
        self,
        train_path: str | Path,
        val_path: str | Path,
        system_prompt: Optional[str] = None,
        max_seq_length: int = 512,
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_seq_length = max_seq_length

    def load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read lines from JSONL file into a list of dictionaries."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")

        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[!] Warning: Skipping invalid JSON on line {line_num} in {file_path}: {e}")
        return records

    def format_records(self, raw_records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Transform raw Alpaca dicts into records containing 'text' field for SFTTrainer."""
        formatted_list = []
        for record in raw_records:
            instruction = record.get("instruction", "")
            context = record.get("input", "")
            output = record.get("output", "")

            formatted_text = format_llama3_chat(
                instruction=instruction,
                context=context,
                output=output,
                system_prompt=self.system_prompt,
            )
            formatted_list.append({"text": formatted_text})
        return formatted_list

    def get_datasets(self) -> tuple[Any, Any]:
        """Load and return formatted Hugging Face Dataset objects (train, val).

        Falls back to raw formatted lists if 'datasets' package is not yet imported.
        """
        raw_train = self.load_jsonl(self.train_path)
        raw_val = self.load_jsonl(self.val_path)

        formatted_train = self.format_records(raw_train)
        formatted_val = self.format_records(raw_val)

        try:
            from datasets import Dataset
            train_dataset = Dataset.from_list(formatted_train)
            val_dataset = Dataset.from_list(formatted_val)
            return train_dataset, val_dataset
        except ImportError:
            # Fallback for environments before pip install datasets
            return formatted_train, formatted_val

