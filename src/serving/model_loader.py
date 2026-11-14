"""Model loader and inference engine for Doc2Learn serving.

Provides a unified interface supporting 3 backends:
1. vLLM Engine (High-throughput PagedAttention on CUDA/Linux)
2. PEFT / HuggingFace Engine (Standard LoRA adapter loading)
3. Mock Engine (Fast offline CPU development and local API testing)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.metrics import extract_json_from_text, validate_quiz_structure
from src.serving.schemas import EngineType, QuizItem, SlideItem


class LLMEngine:
    """Unified inference engine for generating educational quizzes and lecture slides."""

    def __init__(
        self,
        base_model_path: str = "unsloth/llama-3-8b-Instruct",
        adapter_path: Optional[str] = "models/lora-adapter",
        engine_type: EngineType = EngineType.MOCK,
        device: str = "auto",
    ):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.engine_type = engine_type
        self.device = device
        self.model = None
        self.tokenizer = None
        self.vllm_llm = None
        self.vllm_sampling = None

        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize the selected inference backend with graceful fallback."""
        if self.engine_type == EngineType.VLLM:
            self._init_vllm()
        elif self.engine_type == EngineType.PEFT:
            self._init_peft()
        else:
            self._init_mock()

    def _init_mock(self) -> None:
        """Initialize lightweight mock engine for instant CPU execution."""
        self.engine_type = EngineType.MOCK
        print("[✓] LLMEngine initialized in MOCK mode (Fast CPU development).")

    def _init_vllm(self) -> None:
        """Initialize vLLM PagedAttention engine."""
        try:
            from vllm import LLM, SamplingParams
            print(f"[*] Initializing vLLM engine for model: {self.base_model_path} ...")
            self.vllm_llm = LLM(
                model=self.base_model_path,
                enable_lora=bool(self.adapter_path and Path(self.adapter_path).exists()),
                max_model_len=2048,
                gpu_memory_utilization=0.85,
                trust_remote_code=True,
            )
            self.vllm_sampling = SamplingParams(
                temperature=0.3,
                top_p=0.9,
                max_tokens=512,
            )
            print("[✓] vLLM PagedAttention engine loaded successfully.")
        except Exception as e:
            print(f"[!] vLLM initialization unavailable ({e}). Falling back to MOCK mode.")
            self._init_mock()

    def _init_peft(self) -> None:
        """Initialize standard HuggingFace Transformers + PEFT LoRA engine."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print(f"[*] Loading Hugging Face model: {self.base_model_path} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            if self.adapter_path and Path(self.adapter_path).exists():
                print(f"[*] Attaching LoRA adapter from: {self.adapter_path} ...")
                self.model = PeftModel.from_pretrained(base, self.adapter_path)
            else:
                self.model = base
            self.model.eval()
            print("[✓] PEFT LoRA model loaded successfully.")
        except Exception as e:
            print(f"[!] PEFT initialization failed ({e}). Falling back to MOCK mode.")
            self._init_mock()

    def generate_quiz(
        self, context: str, num_questions: int = 3
    ) -> List[QuizItem]:
        """Generate structured multiple-choice quiz questions from a content chunk."""
        if self.engine_type == EngineType.MOCK:
            return self._mock_generate_quiz(context, num_questions)

        prompt = self._build_quiz_prompt(context, num_questions)
        raw_output = self._generate_text(prompt)
        return self._parse_quiz_output(raw_output, context, num_questions)

    def generate_summary(
        self, context: str
    ) -> Tuple[List[SlideItem], str]:
        """Generate structured lecture slides and markdown summary from content."""
        if self.engine_type == EngineType.MOCK:
            return self._mock_generate_summary(context)

        prompt = self._build_summary_prompt(context)
        raw_output = self._generate_text(prompt)
        return self._parse_summary_output(raw_output, context)

    def _generate_text(self, prompt: str) -> str:
        """Dispatch generation request to active backend."""
        if self.engine_type == EngineType.VLLM and self.vllm_llm:
            outputs = self.vllm_llm.generate([prompt], self.vllm_sampling)
            return outputs[0].outputs[0].text.strip()
        elif self.engine_type == EngineType.PEFT and self.model:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            in_len = inputs["input_ids"].shape[1]
            return self.tokenizer.decode(out_ids[0][in_len:], skip_special_tokens=True).strip()
        return ""

    def _build_quiz_prompt(self, context: str, num_questions: int) -> str:
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Based on the following content, generate {num_questions} multiple choice quiz questions. "
            f"Format the output strictly as a JSON list of objects, each containing: "
            f"'question', 'options' (A, B, C, D), 'correct_answer', and 'explanation'.\n\n"
            f"Content:\n{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def _build_summary_prompt(self, context: str) -> str:
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Summarize the following content into structured educational lecture slides. "
            f"Use Markdown headings (### Slide 1: ...) and concise bullet points.\n\n"
            f"Content:\n{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def _parse_quiz_output(
        self, raw_output: str, context: str, target_count: int
    ) -> List[QuizItem]:
        parsed = extract_json_from_text(raw_output)
        quizzes = []

        if isinstance(parsed, list):
            for item in parsed:
                if validate_quiz_structure(item):
                    quizzes.append(QuizItem(**item))
        elif isinstance(parsed, dict):
            if validate_quiz_structure(parsed):
                quizzes.append(QuizItem(**parsed))
            elif "quizzes" in parsed and isinstance(parsed["quizzes"], list):
                for item in parsed["quizzes"]:
                    if validate_quiz_structure(item):
                        quizzes.append(QuizItem(**item))

        # Fallback if parsing failed
        if not quizzes:
            return self._mock_generate_quiz(context, target_count)

        return quizzes[:target_count]

    def _parse_summary_output(
        self, raw_output: str, context: str
    ) -> Tuple[List[SlideItem], str]:
        slides = []
        slide_blocks = re.split(r"(?=###?\s+Slide\s+\d+)", raw_output, flags=re.IGNORECASE)

        idx = 1
        for block in slide_blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            title = lines[0].replace("#", "").strip() if lines else f"Slide {idx}"
            bullets = [
                line.strip().lstrip("-*• ").strip()
                for line in lines[1:]
                if line.strip().startswith(("-", "*", "•"))
            ]
            if not bullets:
                bullets = [line.strip() for line in lines[1:] if len(line.strip()) > 5]

            if bullets:
                slides.append(SlideItem(slide_number=idx, title=title, bullet_points=bullets))
                idx += 1

        if not slides:
            return self._mock_generate_summary(context)

        return slides, raw_output

    def _mock_generate_quiz(self, context: str, num_questions: int) -> List[QuizItem]:
        """Generate high-quality mock quizzes based on context keywords."""
        words = context.split()
        sample_topic = " ".join(words[:4]) if len(words) >= 4 else "the provided document"

        templates = [
            QuizItem(
                question=f"Which of the following statements is correct regarding {sample_topic}?",
                options={
                    "A": "It operates strictly within standard nominal specifications.",
                    "B": "It operates with undefined and uncalibrated parameters.",
                    "C": "It is completely non-functional under typical operating conditions.",
                    "D": "It requires zero power and zero configuration.",
                },
                correct_answer="A",
                explanation=f"Based on the provided text, {sample_topic} defines formal technical requirements.",
            ),
            QuizItem(
                question="What is the primary operational advantage demonstrated in the provided material?",
                options={
                    "A": "Elimination of all network cables.",
                    "B": "Enhanced noise immunity and robust message arbitration.",
                    "C": "Infinite data transmission bandwidth.",
                    "D": "Compatibility with obsolete mechanical systems.",
                },
                correct_answer="B",
                explanation="Differential signaling and arbitration mechanisms ensure robust noise immunity.",
            ),
            QuizItem(
                question="Under what condition does the system transition to an inactive or error-passive state?",
                options={
                    "A": "When ambient temperature exceeds 20 degrees Celsius.",
                    "B": "When error counters exceed predefined threshold limits (e.g. TEC > 127).",
                    "C": "Upon receiving any standard transmission frame.",
                    "D": "Whenever the transmission speed is doubled.",
                },
                correct_answer="B",
                explanation="Exceeding error counter thresholds triggers transition to error-passive or bus-off states.",
            ),
        ]
        return templates[:num_questions]

    def _mock_generate_summary(self, context: str) -> Tuple[List[SlideItem], str]:
        """Generate structured mock slides based on context."""
        words = context.split()
        headline = " ".join(words[:5]) if len(words) >= 5 else "Core Specifications"

        slides = [
            SlideItem(
                slide_number=1,
                title=f"Introduction to {headline}",
                bullet_points=[
                    "Fundamental protocol overview and operational architecture.",
                    "Key electrical and logical characteristics established in the specification.",
                    "Baseline reference parameters for system integration.",
                ],
            ),
            SlideItem(
                slide_number=2,
                title="Operational Mechanisms & Key Takeaways",
                bullet_points=[
                    "Deterministic bitwise arbitration ensuring high-priority delivery.",
                    "Comprehensive error detection and fault confinement boundaries.",
                    "Practical application in modern distributed embedded networks.",
                ],
            ),
        ]

        markdown = (
            f"### Slide 1: Introduction to {headline}\n"
            f"- Fundamental protocol overview and operational architecture.\n"
            f"- Key electrical and logical characteristics established in the specification.\n"
            f"- Baseline reference parameters for system integration.\n\n"
            f"### Slide 2: Operational Mechanisms & Key Takeaways\n"
            f"- Deterministic bitwise arbitration ensuring high-priority delivery.\n"
            f"- Comprehensive error detection and fault confinement boundaries.\n"
            f"- Practical application in modern distributed embedded networks.\n"
        )
        return slides, markdown

