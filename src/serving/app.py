"""FastAPI application for Doc2Learn model serving.

Exposes high-speed RESTful endpoints for:
1. Health & status check (GET /api/health)
2. Raw text & topic generation (POST /api/generate/text)
3. Uploaded document generation (POST /api/generate/file)
4. Web article & YouTube transcript generation (POST /api/generate/url)
"""

from __future__ import annotations

import os
import time
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from src.serving.content_extractor import (
    YOUTUBE_REGEX,
    extract_from_file,
    extract_from_text,
    extract_from_web_url,
    extract_from_youtube_url,
    process_and_chunk_content,
)
from src.serving.model_loader import LLMEngine
from src.serving.schemas import (
    EngineType,
    FullGenerationResponse,
    GenerationMetadata,
    HealthResponse,
    TaskType,
    TextGenerationRequest,
    UrlGenerationRequest,
)

# ---------------------------------------------------------------------------
# App Initialization & Middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Doc2Learn API",
    description="High-Speed Educational Content & Quiz Generation Engine powered by Fine-Tuned Llama-3-8B and vLLM.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration (Streamlit, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine default engine from environment (defaults to MOCK on local CPU)
DEFAULT_ENGINE = os.getenv("DOC2LEARN_ENGINE", "mock").lower()
engine_choice = EngineType.VLLM if DEFAULT_ENGINE == "vllm" else (
    EngineType.PEFT if DEFAULT_ENGINE == "peft" else EngineType.MOCK
)

# Global engine instance
llm_engine = LLMEngine(
    base_model_path=os.getenv("BASE_MODEL_PATH", "unsloth/llama-3-8b-Instruct"),
    adapter_path=os.getenv("ADAPTER_PATH", "models/lora-adapter"),
    engine_type=engine_choice,
)


# ---------------------------------------------------------------------------
# Helper Pipeline Executor
# ---------------------------------------------------------------------------

def _execute_generation_pipeline(
    raw_text: str,
    source_type: str,
    source_name: str,
    task_type: TaskType = TaskType.BOTH,
    num_questions: int = 3,
) -> FullGenerationResponse:
    """Core orchestrator: Chunks content, runs inference, and measures latency."""
    start_time = time.perf_counter()

    # Step 1: Chunk content into 512-token segments
    chunks = process_and_chunk_content(
        raw_text=raw_text,
        source_metadata={"source_type": source_type, "source_name": source_name},
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The provided content could not be segmented into readable chunks.",
        )

    # Use primary chunk content for generation
    primary_context = "\n\n".join(c["text"] for c in chunks[:3])

    quizzes = []
    slides = []
    full_markdown = ""

    # Step 2: Generate Quiz if requested
    if task_type in (TaskType.QUIZ, TaskType.BOTH):
        quizzes = llm_engine.generate_quiz(
            context=primary_context, num_questions=num_questions
        )

    # Step 3: Generate Summary/Slides if requested
    if task_type in (TaskType.SUMMARY, TaskType.BOTH):
        slides, full_markdown = llm_engine.generate_summary(context=primary_context)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    metadata = GenerationMetadata(
        source_type=source_type,
        source_name=source_name,
        total_characters=len(raw_text),
        total_chunks=len(chunks),
        inference_time_ms=latency_ms,
        engine_used=llm_engine.engine_type.value,
    )

    return FullGenerationResponse(
        quizzes=quizzes,
        slides=slides,
        full_markdown=full_markdown,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check server health and active model backend",
)
def health_check() -> HealthResponse:
    """Returns runtime status, GPU detection, and the loaded model engine."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        engine=llm_engine.engine_type,
        base_model=llm_engine.base_model_path,
        adapter_loaded=bool(llm_engine.adapter_path),
        gpu_available=has_gpu,
        device="cuda" if has_gpu else "cpu",
    )


@app.post(
    "/api/generate/text",
    response_model=FullGenerationResponse,
    tags=["Generation"],
    summary="Generate Quiz & Lecture Slides from raw text or a topic description",
)
def generate_from_text(payload: TextGenerationRequest) -> FullGenerationResponse:
    """Accepts pasted text or topic description, segments it, and generates educational content."""
    try:
        raw_text = extract_from_text(payload.text)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return _execute_generation_pipeline(
        raw_text=raw_text,
        source_type="text",
        source_name="pasted_text",
        task_type=payload.task_type,
        num_questions=payload.num_questions,
    )


@app.post(
    "/api/generate/file",
    response_model=FullGenerationResponse,
    tags=["Generation"],
    summary="Upload a PDF, TXT, or Markdown document to generate educational content",
)
async def generate_from_file(
    file: UploadFile = File(..., description="Uploaded document (PDF, TXT, MD)"),
    task_type: TaskType = Form(default=TaskType.BOTH),
    num_questions: int = Form(default=3),
) -> FullGenerationResponse:
    """Accepts document file uploads, extracts readable text, and generates quizzes and slides."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file missing filename."
        )

    file_bytes = await file.read()
    try:
        raw_text = extract_from_file(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return _execute_generation_pipeline(
        raw_text=raw_text,
        source_type=f"file_{file.filename.split('.')[-1].lower()}",
        source_name=file.filename,
        task_type=task_type,
        num_questions=num_questions,
    )


@app.post(
    "/api/generate/url",
    response_model=FullGenerationResponse,
    tags=["Generation"],
    summary="Generate from a web article URL or YouTube video link",
)
def generate_from_url(payload: UrlGenerationRequest) -> FullGenerationResponse:
    """Scrapes a web article or fetches a YouTube video transcript to generate quizzes and slides."""
    url = payload.url.strip()

    try:
        if YOUTUBE_REGEX.search(url):
            raw_text = extract_from_youtube_url(url)
            source_type = "url_youtube"
        else:
            raw_text = extract_from_web_url(url)
            source_type = "url_web"
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return _execute_generation_pipeline(
        raw_text=raw_text,
        source_type=source_type,
        source_name=url,
        task_type=payload.task_type,
        num_questions=payload.num_questions,
    )


def create_app() -> FastAPI:
    """Factory function for testing and production ASGI servers."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=True)
