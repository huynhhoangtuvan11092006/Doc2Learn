"""Orchestrator for reading chunks, generating synthetic instruction samples, filtering, and splitting."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any

from .data_generator import DataGenerator
from .deduplicator import DatasetDeduplicator


class DatasetBuilder:
    """Builds clean, deduplicated, train/validation split datasets from processed chunks."""

    def __init__(
        self,
        generator: DataGenerator,
        deduplicator: DatasetDeduplicator,
        val_split_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        self.generator = generator
        self.deduplicator = deduplicator
        self.val_split_ratio = val_split_ratio
        self.random_seed = random_seed

    def load_chunks(self, chunks_path: str | Path) -> List[Dict[str, Any]]:
        """Load JSONL chunks file."""
        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def build_dataset(
        self,
        chunks_path: str | Path = "data/processed/chunks.jsonl",
        output_dir: str | Path = "data/synthetic",
        task_types: List[str] = None,
    ) -> Dict[str, Any]:
        """Execute the full dataset synthesis pipeline.

        Returns:
            Dictionary of execution statistics.
        """
        task_types = task_types or ["quiz", "summary", "troubleshooting"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chunks = self.load_chunks(chunks_path)
        print(f"[*] Loaded {len(chunks)} chunks from {chunks_path}")

        raw_samples: List[Dict[str, Any]] = []

        print(f"[*] Generating synthetic instruction-following pairs across tasks: {task_types}...")
        for chunk in chunks:
            for task in task_types:
                generated = self.generator.generate_for_chunk(chunk, task_type=task)
                raw_samples.extend(generated)

        print(f"[+] Total raw samples generated: {len(raw_samples)}")

        # Step 2: Quality Filtering & Deduplication
        print("[*] Running deduplication and quality validation...")
        filtered_samples = self.deduplicator.filter_dataset(raw_samples)
        dedup_stats = self.deduplicator.get_summary()
        print(f"[+] Samples accepted: {len(filtered_samples)} | Dropped duplicates: {dedup_stats['dropped_duplicate']} | Dropped malformed: {dedup_stats['dropped_malformed']}")

        # Step 3: Train / Validation Split
        random.seed(self.random_seed)
        random.shuffle(filtered_samples)

        val_size = max(1, int(len(filtered_samples) * self.val_split_ratio))
        val_samples = filtered_samples[:val_size]
        train_samples = filtered_samples[val_size:]

        # Step 4: Write to JSONL
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"[✓] Saved {len(train_samples)} training samples to: {train_path}")
        print(f"[✓] Saved {len(val_samples)} validation samples to: {val_path}")

        summary = {
            "total_chunks_processed": len(chunks),
            "total_raw_generated": len(raw_samples),
            "total_accepted": len(filtered_samples),
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "deduplication_stats": dedup_stats,
            "train_file": str(train_path),
            "val_file": str(val_path),
        }
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doc2Learn Dataset Synthesis Pipeline")
    parser.add_argument("--chunks", type=str, default="data/processed/chunks.jsonl", help="Path to chunks.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--mock", action="store_true", default=True, help="Use deterministic mock generation")

    args = parser.parse_args()

    gen = DataGenerator(use_mock=args.mock)
    dedup = DatasetDeduplicator(similarity_threshold=0.75)
    builder = DatasetBuilder(generator=gen, deduplicator=dedup, val_split_ratio=args.val_split)

    builder.build_dataset(chunks_path=args.chunks, output_dir=args.output_dir)
