"""Evaluation module for Doc2Learn-LLM.

Provides metrics, hallucination detection, inference comparison,
human evaluation, and benchmark reporting for evaluating Base vs. Fine-tuned models.
"""

from src.evaluation.metrics import (
    compute_rouge_l,
    compute_bleu_4,
    compute_format_compliance,
    compute_perplexity,
    evaluate_predictions,
)
from src.evaluation.hallucination_checker import (
    check_hallucination,
    batch_check_hallucination,
    extract_technical_entities,
)
from src.evaluation.inference import (
    run_inference_benchmark,
    load_evaluation_data,
)
from src.evaluation.human_eval import (
    run_human_evaluation,
)
from src.evaluation.report_generator import (
    run_benchmark_pipeline,
    generate_markdown_report,
)

__all__ = [
    "compute_rouge_l",
    "compute_bleu_4",
    "compute_format_compliance",
    "compute_perplexity",
    "evaluate_predictions",
    "check_hallucination",
    "batch_check_hallucination",
    "extract_technical_entities",
    "run_inference_benchmark",
    "load_evaluation_data",
    "run_human_evaluation",
    "run_benchmark_pipeline",
    "generate_markdown_report",
]
