"""Unit and integration tests for Doc2Learn FastAPI serving endpoints."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.content_extractor import (
    extract_from_file,
    extract_from_text,
    process_and_chunk_content,
    YOUTUBE_REGEX,
)
from src.serving.benchmark import run_latency_benchmark, generate_serving_report


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


class TestServingEndpoints:
    """Tests for FastAPI HTTP endpoints."""

    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "engine" in data
        assert "device" in data
        assert "base_model" in data

    def test_generate_text_full(self, client):
        payload = {
            "text": "The Controller Area Network (CAN) protocol uses differential signaling to maintain noise immunity.",
            "task_type": "both",
            "num_questions": 2,
        }
        resp = client.post("/api/generate/text", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "quizzes" in data
        assert "slides" in data
        assert "metadata" in data
        assert len(data["quizzes"]) == 2
        assert len(data["slides"]) >= 1
        assert data["metadata"]["source_type"] == "text"
        assert data["metadata"]["inference_time_ms"] > 0

    def test_generate_text_quiz_only(self, client):
        payload = {
            "text": "CAN protocol utilizes CSMA/CD with non-destructive bitwise arbitration.",
            "task_type": "quiz",
            "num_questions": 1,
        }
        resp = client.post("/api/generate/text", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["quizzes"]) == 1
        assert data["slides"] == []
        assert data["full_markdown"] == ""

    def test_generate_text_summary_only(self, client):
        payload = {
            "text": "CAN protocol utilizes CSMA/CD with non-destructive bitwise arbitration.",
            "task_type": "summary",
        }
        resp = client.post("/api/generate/text", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["quizzes"] == []
        assert len(data["slides"]) >= 1
        assert "### Slide" in data["full_markdown"]

    def test_generate_text_too_short_fails(self, client):
        payload = {"text": "short"}
        resp = client.post("/api/generate/text", json=payload)
        assert resp.status_code == 422  # Pydantic validation error

    def test_generate_file_pdf(self, client):
        sample_pdf_path = Path("data/raw/sample_can_bus.pdf")
        if sample_pdf_path.exists():
            with open(sample_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            resp = client.post(
                "/api/generate/file",
                files={"file": ("sample_can_bus.pdf", pdf_bytes, "application/pdf")},
                data={"task_type": "both", "num_questions": "2"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["quizzes"]) == 2
            assert data["metadata"]["source_type"] == "file_pdf"

    def test_generate_file_markdown(self, client):
        md_content = b"# Topic: Automotive Networks\n\nCAN High is 3.5V and CAN Low is 1.5V in dominant state."
        resp = client.post(
            "/api/generate/file",
            files={"file": ("lesson.md", md_content, "text/markdown")},
            data={"task_type": "both", "num_questions": "1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["quizzes"]) == 1
        assert data["metadata"]["source_type"] == "file_md"

    def test_generate_file_unsupported_extension(self, client):
        resp = client.post(
            "/api/generate/file",
            files={"file": ("malicious.exe", b"binary content", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported file format" in resp.json()["detail"]


class TestContentExtractorAndBenchmark:
    """Unit tests for content extraction helpers and benchmark generation."""

    def test_extract_from_text(self):
        assert extract_from_text("  hello world  ") == "hello world"
        with pytest.raises(ValueError):
            extract_from_text("   ")

    def test_extract_from_file_txt(self):
        content = extract_from_file(b"Sample notes text", "test.txt")
        assert content == "Sample notes text"

    def test_youtube_regex(self):
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        ]
        for u in urls:
            match = YOUTUBE_REGEX.search(u)
            assert match is not None
            assert match.group(1) == "dQw4w9WgXcQ"

    def test_run_latency_benchmark_helper(self, client, tmp_path):
        text = "CAN Bus is a robust vehicle bus standard designed for microcontrollers to communicate."
        stats = run_latency_benchmark(client, text, num_requests=3)
        assert stats["total_requests"] == 3
        assert stats["mean_latency_ms"] > 0
        assert stats["throughput_req_per_sec"] > 0

        report_file = tmp_path / "serving_report.md"
        report_md = generate_serving_report(stats, output_path=report_file)
        assert report_file.exists()
        assert "PagedAttention" in report_md
