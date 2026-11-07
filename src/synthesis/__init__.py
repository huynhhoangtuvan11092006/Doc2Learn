"""Synthesis package for generating synthetic instruction-tuning datasets from technical chunks."""

from .prompt_templates import (
    QuizQuestion,
    QuizResponse,
    SummaryResponse,
    TroubleshootingResponse,
    PROMPT_REGISTRY,
)
from .deduplicator import DatasetDeduplicator
from .data_generator import DataGenerator
from .dataset_builder import DatasetBuilder

__all__ = [
    "QuizQuestion",
    "QuizResponse",
    "SummaryResponse",
    "TroubleshootingResponse",
    "PROMPT_REGISTRY",
    "DatasetDeduplicator",
    "DataGenerator",
    "DatasetBuilder",
]
