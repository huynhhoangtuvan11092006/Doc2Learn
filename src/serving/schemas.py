"""Pydantic schemas for the Doc2Learn FastAPI serving endpoints.

Defines strict request/response data contracts for:
- Text & URL inputs
- Structured Quiz items (4 options, correct answer, explanation)
- Structured Slide outlines (Slide number, title, bullet points)
- Comprehensive generation responses and system health check.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl


class TaskType(str, Enum):
    """Allowed generation task modes."""
    QUIZ = "quiz"
    SUMMARY = "summary"
    BOTH = "both"


class EngineType(str, Enum):
    """Supported model serving backends."""
    VLLM = "vllm"
    PEFT = "peft"
    MOCK = "mock"


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class TextGenerationRequest(BaseModel):
    """Request payload for raw text / pasted content input."""
    text: str = Field(
        ...,
        min_length=10,
        description="Raw technical or educational text content to generate from.",
        examples=["In CAN bus, the recessive state has both lines at 2.5V, while dominant state drives CAN-H to 3.5V and CAN-L to 1.5V."]
    )
    task_type: TaskType = Field(
        default=TaskType.BOTH,
        description="Type of content to generate ('quiz', 'summary', or 'both')."
    )
    num_questions: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Target number of quiz questions to generate (1 to 20)."
    )
    target_audience: Optional[str] = Field(
        default="engineering_students",
        description="Target learner level (e.g. 'beginner', 'engineering_students', 'professional')."
    )


class UrlGenerationRequest(BaseModel):
    """Request payload for web URL or YouTube video link input."""
    url: str = Field(
        ...,
        description="Full URL to a web article or YouTube video.",
        examples=["https://en.wikipedia.org/wiki/CAN_bus", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    )
    task_type: TaskType = Field(
        default=TaskType.BOTH,
        description="Type of content to generate ('quiz', 'summary', or 'both')."
    )
    num_questions: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Target number of quiz questions to generate (1 to 20)."
    )


# ---------------------------------------------------------------------------
# Content Entity Schemas
# ---------------------------------------------------------------------------

class QuizItem(BaseModel):
    """Individual multiple-choice question item adhering to strict schema."""
    question: str = Field(..., description="The multiple choice question prompt.")
    options: Dict[str, str] = Field(
        ...,
        description="Dictionary of 4 answer options with keys 'A', 'B', 'C', 'D'."
    )
    correct_answer: str = Field(
        ...,
        description="Key of the correct option ('A', 'B', 'C', or 'D')."
    )
    explanation: Optional[str] = Field(
        default="",
        description="Detailed pedagogical explanation citing technical reasons."
    )


class SlideItem(BaseModel):
    """Individual lecture slide outline item."""
    slide_number: int = Field(..., ge=1, description="Sequential slide index.")
    title: str = Field(..., description="Concise slide headline / topic.")
    bullet_points: List[str] = Field(
        ...,
        min_length=1,
        description="Core conceptual bullet points and key takeaways."
    )


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class GenerationMetadata(BaseModel):
    """Execution telemetry and content metrics."""
    source_type: str = Field(..., description="Source origin: 'text', 'file_pdf', 'url_web', 'url_youtube'.")
    source_name: str = Field(default="", description="Original file name or URL if applicable.")
    total_characters: int = Field(..., description="Length of processed input text.")
    total_chunks: int = Field(..., description="Number of 512-token semantic chunks created.")
    inference_time_ms: float = Field(..., description="Total processing time in milliseconds.")
    engine_used: str = Field(..., description="Inference engine backend ('vllm', 'peft', or 'mock').")


class QuizResponse(BaseModel):
    """Response payload containing generated quizzes."""
    quizzes: List[QuizItem] = Field(..., description="List of generated multiple choice questions.")
    metadata: GenerationMetadata


class SummaryResponse(BaseModel):
    """Response payload containing generated lecture summary/slides."""
    slides: List[SlideItem] = Field(..., description="Structured slide outlines.")
    full_markdown: str = Field(..., description="Raw Markdown formatted summary text.")
    metadata: GenerationMetadata


class FullGenerationResponse(BaseModel):
    """Comprehensive response payload containing both Quiz and Summary."""
    quizzes: List[QuizItem] = Field(default_factory=list, description="Generated quizzes.")
    slides: List[SlideItem] = Field(default_factory=list, description="Structured slide outlines.")
    full_markdown: str = Field(default="", description="Complete Markdown summary.")
    metadata: GenerationMetadata


class HealthResponse(BaseModel):
    """System health and model engine status."""
    status: str = Field(default="healthy", description="Service status: 'healthy', 'degraded', or 'offline'.")
    version: str = Field(default="1.0.0", description="Doc2Learn API version.")
    engine: EngineType = Field(..., description="Active inference engine backend.")
    base_model: str = Field(..., description="Underlying base LLM identifier.")
    adapter_loaded: bool = Field(..., description="Whether fine-tuned LoRA weights are loaded.")
    gpu_available: bool = Field(..., description="Whether CUDA GPU is detected and active.")
    device: str = Field(..., description="Compute device: 'cuda', 'cpu', etc.")

