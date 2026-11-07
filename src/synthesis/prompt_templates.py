"""Prompt templates and Pydantic schemas for synthetic instruction-tuning dataset generation."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Output Validation
# ---------------------------------------------------------------------------

class QuizQuestion(BaseModel):
    """Schema for a single technical multiple-choice question."""
    question: str = Field(description="Clear, unambiguous technical question based strictly on the text.")
    options: Dict[Literal["A", "B", "C", "D"], str] = Field(
        description="Dictionary with keys 'A', 'B', 'C', 'D' representing plausible choices."
    )
    correct_answer: Literal["A", "B", "C", "D"] = Field(
        description="The single correct option key."
    )
    explanation: str = Field(
        description="Technical justification for why the correct option is right and others are wrong."
    )
    citation: str = Field(
        description="Explicit reference to the source section, figure, or table in the context."
    )


class QuizResponse(BaseModel):
    """Collection of quiz questions generated from a single chunk."""
    quizzes: List[QuizQuestion] = Field(min_length=1, max_length=5)


class SummaryResponse(BaseModel):
    """Schema for a concise technical lecture slide outline."""
    title: str = Field(description="Core technical topic title.")
    key_concepts: List[str] = Field(min_length=2, max_length=6, description="Bullet points of fundamental concepts.")
    technical_specs: Dict[str, str] = Field(
        default_factory=dict,
        description="Key numerical parameters, voltages, thresholds, or timings found in the text."
    )
    takeaway: str = Field(description="One-sentence essential takeaway for an engineering student.")


class TroubleshootingResponse(BaseModel):
    """Schema for a technical diagnosis / troubleshooting case study."""
    scenario: str = Field(description="Realistic industrial / engineering fault condition.")
    symptom: str = Field(description="Observed anomaly or telemetry indicator.")
    root_cause: str = Field(description="Underlying technical cause determined from the document.")
    remedy: str = Field(description="Actionable corrective step according to technical specifications.")
    safety_warning: Optional[str] = Field(default=None, description="Relevant safety precaution.")


# ---------------------------------------------------------------------------
# Prompt Templates with Strict Negative Constraints (Grounded Generation)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are a Principal Curriculum Engineer and Technical Evaluator for an advanced Digital Engineering Academy.
Your mission is to generate high-rigor, pedagogical training materials from technical specifications, datasheets, and engineering standards.

STRICT GROUNDING RULES:
1. Grounding: All questions, answers, and summaries MUST be directly verifiable from the provided context. Never invent, extrapolate, or introduce external facts.
2. Plausible Distractors: For multiple-choice questions, incorrect options MUST represent realistic technical misconceptions (e.g., reversed polarity, wrong threshold, confused registers). Never use trivial options like 'All of the above' or 'None of the above'.
3. Citation: Always include exact textual citations (section numbers, headings, or parameter names).
4. Output Format: Return strictly valid JSON complying with the requested schema. No markdown conversational filler.
"""

QUIZ_TASK_INSTRUCTION = """Generate 2 to 3 rigorous technical multiple-choice questions based ONLY on the following technical text.

CONTEXT:
{context}

SCHEMA REQUIRED:
{{
  "quizzes": [
    {{
      "question": "string",
      "options": {{
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string"
      }},
      "correct_answer": "A" | "B" | "C" | "D",
      "explanation": "string",
      "citation": "string"
    }}
  ]
}}
"""

SUMMARY_TASK_INSTRUCTION = """Synthesize a structured lecture summary and technical parameter table from the following technical text.

CONTEXT:
{context}

SCHEMA REQUIRED:
{{
  "title": "string",
  "key_concepts": ["bullet 1", "bullet 2"],
  "technical_specs": {{"parameter_name": "value with unit"}},
  "takeaway": "string"
}}
"""

TROUBLESHOOTING_TASK_INSTRUCTION = """Construct a realistic fault diagnosis / troubleshooting case study derived from the failure modes or limits described in the text.

CONTEXT:
{context}

SCHEMA REQUIRED:
{{
  "scenario": "string",
  "symptom": "string",
  "root_cause": "string",
  "remedy": "string",
  "safety_warning": "string"
}}
"""

PROMPT_REGISTRY = {
    "quiz": {
        "system_prompt": SYSTEM_PROMPT_BASE,
        "task_instruction": QUIZ_TASK_INSTRUCTION,
        "response_model": QuizResponse,
        "alpaca_instruction": (
            "Dựa vào đoạn văn bản kỹ thuật được cung cấp, hãy biên soạn các câu hỏi trắc nghiệm chuyên sâu "
            "gồm 4 lựa chọn A, B, C, D, chỉ rõ đáp án chính xác, lời giải thích chi tiết và trích dẫn mục tài liệu."
        ),
    },
    "summary": {
        "system_prompt": SYSTEM_PROMPT_BASE,
        "task_instruction": SUMMARY_TASK_INSTRUCTION,
        "response_model": SummaryResponse,
        "alpaca_instruction": (
            "Hãy tóm tắt đoạn tài liệu kỹ thuật sau thành cấu trúc bài giảng gồm các khái niệm cốt lõi, "
            "bảng thông số kỹ thuật (nếu có) và thông điệp quan trọng nhất."
        ),
    },
    "troubleshooting": {
        "system_prompt": SYSTEM_PROMPT_BASE,
        "task_instruction": TROUBLESHOOTING_TASK_INSTRUCTION,
        "response_model": TroubleshootingResponse,
        "alpaca_instruction": (
            "Dựa vào các giới hạn kỹ thuật hoặc điều kiện lỗi trong tài liệu, hãy xây dựng một tình huống "
            "xử lý sự cố thực tế gồm triệu chứng, nguyên nhân gốc rễ và biện pháp khắc phục."
        ),
    },
}
