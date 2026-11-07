"""LLM-backed generation engine for transforming technical chunks into validated Alpaca-format training pairs."""

import json
import os
import re
import time
from typing import Dict, List, Any, Optional
import requests

from .prompt_templates import PROMPT_REGISTRY, QuizResponse, SummaryResponse, TroubleshootingResponse


class DataGenerator:
    """Generates structured pedagogical items (quizzes, summaries, case studies) from text chunks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        use_mock: bool = False,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        """Args:

        api_key: API key (OpenAI or Gemini). If None, reads from env var
        OPENAI_API_KEY / GEMINI_API_KEY. api_base: Base URL for OpenAI-compatible
        endpoint (e.g. https://api.openai.com/v1). model_name: Target model
        identifier. use_mock: If True, uses deterministic offline heuristic
        generator (no network required). max_retries: Retry attempts for rate
        limits. backoff_factor: Multiplier for exponential backoff sleep.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model_name = model_name
        self.use_mock = use_mock or (not self.api_key)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def generate_for_chunk(self, chunk: Dict[str, Any], task_type: str = "quiz") -> List[Dict[str, Any]]:
        """Generate Alpaca-format instruction-input-output samples for a given chunk.

        Args:
            chunk: Dictionary containing 'text', 'source_file', 'page_number', 'chunk_id'.
            task_type: One of 'quiz', 'summary', 'troubleshooting'.

        Returns:
            List of Alpaca-format dictionaries: {"instruction": ..., "input": ..., "output": ...}
        """
        if task_type not in PROMPT_REGISTRY:
            raise ValueError(f"Unknown task_type: '{task_type}'. Valid choices: {list(PROMPT_REGISTRY.keys())}")

        task_config = PROMPT_REGISTRY[task_type]
        context = chunk.get("text", "")
        if not context.strip():
            return []

        if self.use_mock:
            raw_response = self._mock_generate(context, task_type)
        else:
            raw_response = self._call_llm_with_retry(context, task_config)

        # Parse and format into Alpaca training pairs
        samples = self._format_alpaca_samples(raw_response, context, chunk, task_type, task_config)
        return samples

    def _call_llm_with_retry(self, context: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Call LLM API with exponential backoff on rate limits (HTTP 429 / 5xx)."""
        system_prompt = task_config["system_prompt"]
        task_instruction = task_config["task_instruction"].format(context=context)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_instruction},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        }

        url = f"{self.api_base.rstrip('/')}/chat/completions"
        delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=45)
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return json.loads(content)
                elif response.status_code == 429:
                    time.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    response.raise_for_status()
            except Exception as e:
                if attempt == self.max_retries:
                    # Fallback to mock generation if network fails continuously
                    return self._mock_generate(context, "quiz")
                time.sleep(delay)
                delay *= self.backoff_factor

        return self._mock_generate(context, "quiz")

    def _mock_generate(self, context: str, task_type: str) -> Dict[str, Any]:
        """Offline deterministic heuristic generator for testing and air-gapped runs."""
        lines = [l.strip() for l in context.split("\n") if l.strip()]
        first_line = lines[0] if lines else "Technical Parameter"

        # Search for numerical values, voltages, registers in context
        numbers = re.findall(r"(\d+(?:\.\d+)?\s*(?:V|ms|Ohm|bytes|kHz|MHz|bit|%|REC|TEC))", context)
        spec = numbers[0] if numbers else "Specified Threshold"

        if task_type == "quiz":
            return {
                "quizzes": [
                    {
                        "question": f"Theo tài liệu kỹ thuật về '{first_line[:60]}', giá trị hoặc điều kiện nào sau đây là chính xác?",
                        "options": {
                            "A": f"Giá trị hoạt động tiêu chuẩn là {spec}",
                            "B": "Giá trị không xác định và dao động tự do",
                            "C": "Giá trị luôn bằng 0 ở mọi điều kiện vận hành",
                            "D": "Không yêu cầu bất kỳ ngưỡng điện áp nào",
                        },
                        "correct_answer": "A",
                        "explanation": f"Tài liệu nêu rõ thông số kỹ thuật chuẩn là {spec}. Các phương án B, C, D đều vi phạm quy chuẩn thiết kế.",
                        "citation": f"Trích dẫn từ mục: '{first_line[:50]}'",
                    }
                ]
            }
        elif task_type == "summary":
            return {
                "title": first_line[:60],
                "key_concepts": [
                    f"Đặc tính kỹ thuật chính: {first_line[:50]}",
                    f"Thông số vận hành được kiểm soát: {spec}",
                ],
                "technical_specs": {"Primary Parameter": spec},
                "takeaway": f"Hệ thống yêu cầu tuân thủ nghiêm ngặt ngưỡng {spec} để duy trì độ tin cậy.",
            }
        else:  # troubleshooting
            return {
                "scenario": f"Sự cố suy giảm tín hiệu hoặc vượt ngưỡng trong khối {first_line[:40]}.",
                "symptom": f"Báo cáo cảnh báo khi thông số lệch khỏi mốc {spec}.",
                "root_cause": "Trở kháng đường truyền không phù hợp hoặc nhiễu điện từ vượt quá giới hạn cho phép.",
                "remedy": f"Kiểm tra lại điện trở đầu cuối và hiệu chỉnh lại về mức tiêu chuẩn {spec}.",
                "safety_warning": "Ngắt nguồn trước khi đo kiểm trở kháng để tránh gây hỏng vi mạch.",
            }

    def _format_alpaca_samples(
        self,
        raw_response: Dict[str, Any],
        context: str,
        chunk: Dict[str, Any],
        task_type: str,
        task_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Validate with Pydantic and format into standard Alpaca instruction-input-output dictionaries."""
        samples = []
        alpaca_instruction = task_config["alpaca_instruction"]
        response_model = task_config["response_model"]

        try:
            validated = response_model.model_validate(raw_response)
        except Exception:
            # If schema validation fails, skip or return empty
            return []

        if task_type == "quiz" and isinstance(validated, QuizResponse):
            for q in validated.quizzes:
                output_str = (
                    f"Câu hỏi: {q.question}\n"
                    f"A. {q.options['A']}\n"
                    f"B. {q.options['B']}\n"
                    f"C. {q.options['C']}\n"
                    f"D. {q.options['D']}\n\n"
                    f"Đáp án đúng: {q.correct_answer}\n"
                    f"Giải thích: {q.explanation}\n"
                    f"Trích dẫn nguồn: {q.citation}"
                )
                samples.append({
                    "instruction": alpaca_instruction,
                    "input": context,
                    "output": output_str,
                    "metadata": {
                        "task_type": "quiz",
                        "source_file": chunk.get("source_file", ""),
                        "page_number": chunk.get("page_number", 0),
                        "chunk_id": chunk.get("chunk_id", 0),
                    },
                })
        elif task_type == "summary" and isinstance(validated, SummaryResponse):
            specs_str = "\n".join(f"- {k}: {v}" for k, v in validated.technical_specs.items())
            concepts_str = "\n".join(f"• {c}" for c in validated.key_concepts)
            output_str = (
                f"### {validated.title}\n\n"
                f"**Khái niệm cốt lõi:**\n{concepts_str}\n\n"
                f"**Thông số kỹ thuật:**\n{specs_str or 'Không có thông số đo lường'}\n\n"
                f"**Điểm cốt lõi cần nhớ:** {validated.takeaway}"
            )
            samples.append({
                "instruction": alpaca_instruction,
                "input": context,
                "output": output_str,
                "metadata": {
                    "task_type": "summary",
                    "source_file": chunk.get("source_file", ""),
                    "page_number": chunk.get("page_number", 0),
                    "chunk_id": chunk.get("chunk_id", 0),
                },
            })
        elif task_type == "troubleshooting" and isinstance(validated, TroubleshootingResponse):
            output_str = (
                f"**Tình huống sự cố:** {validated.scenario}\n"
                f"**Triệu chứng:** {validated.symptom}\n"
                f"**Nguyên nhân gốc rễ:** {validated.root_cause}\n"
                f"**Biện pháp xử lý:** {validated.remedy}\n"
                f"**Cảnh báo an toàn:** {validated.safety_warning or 'Tuân thủ quy trình an toàn điện chung.'}"
            )
            samples.append({
                "instruction": alpaca_instruction,
                "input": context,
                "output": output_str,
                "metadata": {
                    "task_type": "troubleshooting",
                    "source_file": chunk.get("source_file", ""),
                    "page_number": chunk.get("page_number", 0),
                    "chunk_id": chunk.get("chunk_id", 0),
                },
            })

        return samples
