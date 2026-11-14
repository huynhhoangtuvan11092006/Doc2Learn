"""Serving module for Doc2Learn-LLM.

Provides multi-source content extraction (PDF, Text, Web URL, YouTube),
FastAPI endpoints, model loading engines (vLLM, PEFT, Mock), and latency benchmarking.
"""

from src.serving.content_extractor import (
    extract_from_text,
    extract_from_file,
    extract_from_web_url,
    extract_from_youtube_url,
    process_and_chunk_content,
)
from src.serving.schemas import (
    TaskType,
    EngineType,
    TextGenerationRequest,
    UrlGenerationRequest,
    QuizItem,
    SlideItem,
    GenerationMetadata,
    QuizResponse,
    SummaryResponse,
    FullGenerationResponse,
    HealthResponse,
)
from src.serving.model_loader import LLMEngine
from src.serving.app import app, create_app

__all__ = [
    "extract_from_text",
    "extract_from_file",
    "extract_from_web_url",
    "extract_from_youtube_url",
    "process_and_chunk_content",
    "TaskType",
    "EngineType",
    "TextGenerationRequest",
    "UrlGenerationRequest",
    "QuizItem",
    "SlideItem",
    "GenerationMetadata",
    "QuizResponse",
    "SummaryResponse",
    "FullGenerationResponse",
    "HealthResponse",
    "LLMEngine",
    "app",
    "create_app",
]
