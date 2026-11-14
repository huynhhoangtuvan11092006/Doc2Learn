"""Script to download and format allenai/sciq dataset into Doc2Learn standard Alpaca format.

Uses direct HTTPS REST requests via httpx to bypass Windows AppLocker DLL constraints.
Fetches high-quality technical/STEM samples:
- data/synthetic/train.jsonl (800 samples)
- data/synthetic/val.jsonl (100 samples)
- data/processed/chunks.jsonl (corresponding processed chunks with token metadata)
"""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List
import httpx
import tiktoken


HF_ROWS_API = "https://datasets-server.huggingface.co/rows"


def fetch_sciq_rows(split: str, target_count: int) -> List[Dict[str, Any]]:
    """Fetch rows from Hugging Face Datasets Server via paginated REST API."""
    rows: List[Dict[str, Any]] = []
    offset = 0
    limit = 100

    print(f"[*] Fetching {target_count} samples for split '{split}' from Hugging Face API...")
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        while len(rows) < target_count:
            url = f"{HF_ROWS_API}?dataset=allenai%2Fsciq&config=default&split={split}&offset={offset}&limit={limit}"
            try:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
                fetched = [item["row"] for item in data.get("rows", [])]
                if not fetched:
                    break

                for r in fetched:
                    # Filter: ensure support has meaningful length (>= 15 words)
                    if len(r.get("support", "").split()) >= 15:
                        rows.append(r)
                        if len(rows) >= target_count:
                            break

                offset += limit
                print(f"    - Collected {len(rows)}/{target_count} samples...")
                time.sleep(0.2)  # Avoid rate limit
            except Exception as e:
                print(f"[!] Warning during fetch at offset {offset}: {e}")
                time.sleep(1.0)
                if offset > 1000:
                    break

    return rows


def format_sciq_sample(item: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
    """Transform a SciQ record into standard Alpaca Instruction format with randomized option keys."""
    support = item.get("support", "").strip()
    question = item.get("question", "").strip()
    correct_val = item.get("correct_answer", "").strip()
    distractors = [
        item.get("distractor1", "").strip(),
        item.get("distractor2", "").strip(),
        item.get("distractor3", "").strip(),
    ]

    # Combine and shuffle 4 options
    all_options = [correct_val] + distractors
    random.shuffle(all_options)

    keys = ["A", "B", "C", "D"]
    opt_dict = {k: opt for k, opt in zip(keys, all_options)}
    correct_key = next(k for k, v in opt_dict.items() if v == correct_val)

    output_text = (
        f"Câu hỏi: {question}\n"
        f"A. {opt_dict['A']}\n"
        f"B. {opt_dict['B']}\n"
        f"C. {opt_dict['C']}\n"
        f"D. {opt_dict['D']}\n\n"
        f"Đáp án đúng: {correct_key}\n"
        f"Giải thích: Theo nội dung tài liệu, '{correct_val}' là đáp án chính xác dựa trên các luận điểm khoa học được mô tả."
    )

    return {
        "id": sample_id,
        "instruction": (
            "Dựa vào đoạn văn bản kỹ thuật được cung cấp, hãy biên soạn các câu hỏi trắc nghiệm chuyên sâu "
            "gồm 4 lựa chọn A, B, C, D, chỉ rõ đáp án chính xác, lời giải thích chi tiết và trích dẫn mục tài liệu."
        ),
        "input": support,
        "output": output_text,
    }


def main():
    random.seed(42)

    train_rows = fetch_sciq_rows(split="train", target_count=500)
    val_rows = fetch_sciq_rows(split="validation", target_count=60)

    train_out = Path("data/synthetic/train.jsonl")
    val_out = Path("data/synthetic/val.jsonl")
    chunks_out = Path("data/processed/chunks.jsonl")

    train_out.parent.mkdir(parents=True, exist_ok=True)
    chunks_out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Write train.jsonl & chunks.jsonl
    print(f"[*] Writing {len(train_rows)} samples to {train_out} ...")
    chunks = []
    with open(train_out, "w", encoding="utf-8") as f:
        for i, item in enumerate(train_rows):
            alpaca_item = format_sciq_sample(item, sample_id=i)
            f.write(json.dumps(alpaca_item, ensure_ascii=False) + "\n")

            support_text = alpaca_item["input"]
            tokens = len(tokenizer.encode(support_text, disallowed_special=()))
            chunks.append({
                "chunk_id": f"sciq_train_{i}",
                "doc_name": "sciq_scientific_collection",
                "text": support_text,
                "token_count": tokens,
            })

    # Write val.jsonl
    print(f"[*] Writing {len(val_rows)} samples to {val_out} ...")
    with open(val_out, "w", encoding="utf-8") as f:
        for i, item in enumerate(val_rows):
            alpaca_item = format_sciq_sample(item, sample_id=len(train_rows) + i)
            f.write(json.dumps(alpaca_item, ensure_ascii=False) + "\n")

    # Write chunks.jsonl
    print(f"[*] Writing {len(chunks)} chunks to {chunks_out} ...")
    with open(chunks_out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n[✓] Scale-Up SciQ Dataset successfully generated!")
    print(f"    - Train dataset:  {train_out} ({len(train_rows)} samples)")
    print(f"    - Val dataset:    {val_out} ({len(val_rows)} samples)")
    print(f"    - Processed Chunks: {chunks_out} ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
