# report_extractor/extraction/vlm_engine.py

import io
import json
import logging
import re
import ollama
from PIL import Image
from typing import Type
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Compact expected-output example so the model knows the shape
_EXAMPLE_OUTPUT = """{
  "patient": {"name": "John Doe", "age": 45, "gender": "Male", "report_date": "2024-03-15"},
  "lab_tests": [
    {"test_name": "Hemoglobin", "value": "13.5", "unit": "g/dL", "reference_range": "13.0-17.0", "abnormal_flag": false}
  ],
  "diagnosis": ["Iron deficiency anemia"],
  "notes": "Follow-up in 4 weeks."
}"""

_SYSTEM_PROMPT = (
    "You are a medical data extraction assistant. "
    "You will be given an image of a medical / lab report. "
    "Extract every piece of information visible in the image and return it as a JSON object. "
    "Be thorough — do not skip any lab test rows visible in the report. "
    "Use null only when a field truly cannot be found in the image."
)


class VLMEngine:
    """
    Vision-Language extraction engine using local Ollama model.

    Key design choices
    ------------------
    * Uses Ollama's ``format`` parameter with the Pydantic JSON-schema so that
      generation is **constrained** to valid, schema-conformant JSON — this
      eliminates most null-field and malformed-output issues.
    * Keeps the user prompt short and focused so the small model spends its
      capacity on *reading the image* rather than parsing instructions.
    * Separates system / user roles for cleaner context.
    * Retries once with a nudge if the first attempt is mostly nulls.
    """

    def __init__(
        self,
        model_name: str = "thiagomoraes/medgemma-4b-it:Q8_0",
        temperature: float = 0.1,
        max_retries: int = 2,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        image: Image.Image,
        schema: Type[BaseModel],
        system_prompt: str | None = None,
    ) -> dict:
        """
        Extract structured medical report data from *image*.

        Args:
            image:         Preprocessed PIL image.
            schema:        Pydantic model class whose JSON schema drives output.
            system_prompt: Optional override for the system message.

        Returns:
            dict conforming to *schema*.
        """
        image_bytes = self._image_to_bytes(image)
        json_schema = schema.model_json_schema()
        sys_msg = system_prompt or _SYSTEM_PROMPT

        user_prompt = self._build_user_prompt(json_schema)

        for attempt in range(1, self.max_retries + 1):
            logger.info("VLM extraction attempt %d/%d", attempt, self.max_retries)

            parsed = self._call_ollama(
                sys_msg, user_prompt, image_bytes, json_schema
            )

            if not self._is_mostly_null(parsed):
                return parsed

            logger.warning(
                "Attempt %d returned mostly null values — retrying with nudge.",
                attempt,
            )
            # On retry, add an explicit nudge
            user_prompt = self._build_retry_prompt(json_schema)

        # Return whatever we got on the last attempt
        logger.warning("Returning best-effort result after %d attempts.", self.max_retries)
        return parsed

    # ------------------------------------------------------------------ #
    #  INTERNALS                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _build_user_prompt(json_schema: dict) -> str:
        """Short, focused prompt — keeps model attention on the image."""
        return (
            "Look at this medical report image carefully.\n"
            "Extract ALL patient information, lab test results, diagnoses, and notes.\n\n"
            "Rules:\n"
            "- Read every row in every table in the image.\n"
            "- Preserve exact values, units, and reference ranges as printed.\n"
            "- For dates use YYYY-MM-DD format.\n"
            "- Use null ONLY when a field is truly absent from the image.\n"
            "- abnormal_flag: true if marked abnormal (H/L/*/↑/↓), false otherwise.\n\n"
            f"Example output shape:\n{_EXAMPLE_OUTPUT}\n\n"
            "Now extract from the image above. Return ONLY the JSON."
        )

    @staticmethod
    def _build_retry_prompt(json_schema: dict) -> str:
        return (
            "The previous extraction had too many null values. "
            "Please look at the medical report image again MORE CAREFULLY.\n\n"
            "- Read EVERY line of text in the image.\n"
            "- Extract the patient name, age, gender, date.\n"
            "- Extract ALL lab tests with their values, units, and reference ranges.\n"
            "- Do NOT leave fields as null if you can see the information in the image.\n\n"
            f"Expected output shape:\n{_EXAMPLE_OUTPUT}\n\n"
            "Return ONLY the JSON."
        )

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        json_schema: dict,
    ) -> dict:
        """Single Ollama round-trip with structured-output constraint."""

        response = ollama.chat(
            model=self.model_name,
            format=json_schema,          # ← constrained structured output
            options={
                "temperature": self.temperature,
                "num_predict": 4096,
            },
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_bytes],
                },
            ],
        )

        content = response["message"]["content"]
        logger.debug("Raw VLM response:\n%s", content)

        content = self._strip_markdown_fences(content)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from model:\n{content}")

        return parsed

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove ```json … ``` wrappers if present."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    @staticmethod
    def _is_mostly_null(data: dict, threshold: float = 0.7) -> bool:
        """Return True if ≥ threshold of top-level values are None / empty."""
        if not data:
            return True
        values = list(data.values())
        null_count = sum(
            1 for v in values
            if v is None or v == [] or v == {} or v == ""
        )
        return (null_count / len(values)) >= threshold
