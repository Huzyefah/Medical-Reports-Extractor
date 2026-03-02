# midas_extraction/extraction/vlm_engine.py

import json
import ollama
from PIL import Image
from typing import Type
from pydantic import BaseModel


class VLMEngine:
    """
    Vision-Language extraction engine using local Ollama thiagomoraes/medgemma-4b-it:Q8_0.

    This module:
    - sends image + prompt to Ollama
    - receives structured JSON
    - returns parsed dict
    """

    def __init__(
        self,
        model_name: str = "thiagomoraes/medgemma-4b-it:Q8_0",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature

    def extract(
        self,
        image: Image.Image,
        schema: Type[BaseModel],
        system_prompt: str | None = None,
    ) -> dict:
        """
        Extract structured medical report.

        Args:
            image: Preprocessed PIL image
            schema: Pydantic schema class
            system_prompt: optional system prompt

        Returns:
            dict matching schema
        """

        # Convert schema to JSON schema string
        json_schema = schema.model_json_schema()

        prompt = f"""
MEDICAL REPORT EXTRACTION TASK

You are a highly specialized medical data extraction system. Your task is to extract structured clinical information from a medical report image with absolute accuracy and completeness.

CRITICAL INSTRUCTIONS:
1. Extract ONLY information that you can clearly see in the report. Do NOT invent or assume data.
2. If a field is not present or unclear in the report, set it to null (do not use empty strings or made-up values).
3. Preserve EXACT values as they appear - do not normalize or reformat unless specified in field descriptions.
4. For numeric lab values, preserve significant figures and decimal precision exactly as shown.
5. For categorical results (Negative, Positive, Normal, Abnormal), use the EXACT text from the report.

EXTRACTION RULES BY FIELD:

Patient Information:
- name: Extract EXACTLY as written on the report. Include full name if available.
- age: Extract only the numeric value in years. If DOB is given but not age, calculate the age.
- gender: Use 'M', 'F', or exact text as shown (Male, Female). Use null if ambiguous.
- report_date: Extract the date the test was performed/report issued. Format as YYYY-MM-DD. Common labels: "Date of Test", "Report Date", "Specimen Date", "Date Collected", "Date of Service".

Lab Tests (Complete ALL fields for each test):
- test_name: Extract the precise lab test name as shown in the report. Do not abbreviate or expand abbreviations unless needed for clarity.
- value: Extract the numeric or categorical result EXACTLY as displayed. Preserve all precision and decimals. Include negative signs if present.
- unit: Extract the unit of measurement if present (g/dL, mg/dL, 10^3/μL, etc.). Set to null if no unit displayed.
- reference_range: Extract the reference/normal range if present (e.g., "13.5-17.5", "70-100 mg/dL"). Look for columns labeled "Reference Range", "Normal Range", "Ref. Range", "Normal Values". Set to null if not provided.
- abnormal_flag: Set to true ONLY if the report explicitly marks the value as abnormal using symbols (*, ↑, ↓, H, L, HIGH, LOW) or color coding. Otherwise set to false. Set to null if unclear.

Diagnosis Section:
- Extract ONLY clinical diagnoses, impressions, and conclusions - NOT individual lab findings.
- Common section headers: "Diagnosis", "Impression", "Clinical Summary", "Conclusion", "Assessment".
- Return as a list of distinct diagnoses, one per item.
- Use exact medical terminology from the report.

Notes/Recommendations:
- Extract clinical notes, recommendations, or physician comments.
- Common section headers: "Notes", "Recommendations", "Comments", "Remarks", "Follow-up", "Clinical Notes".
- Capture physician instructions or follow-up orders if present.

OUTPUT REQUIREMENTS:
- Return ONLY a single, valid JSON object matching this exact schema:

{json.dumps(json_schema, indent=2)}

- No explanations before or after the JSON
- No markdown code blocks
- No additional text whatsoever
- All field names and types must match the schema exactly
- Use null for missing/unclear fields (not empty strings or false)
- Use arrays [] for list fields with no data (not null)
- Ensure all dates are in YYYY-MM-DD format
- Double-check JSON validity before returning
"""

        if system_prompt:
            prompt = system_prompt + "\n\n" + prompt

        # Convert PIL image to bytes
        import io
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        # Call Ollama
        response = ollama.chat(
            model=self.model_name,
            options={
                "temperature": self.temperature,
            },
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }
            ],
        )

        content = response["message"]["content"]

        # Clean up the response - remove markdown code blocks
        content = content.strip()
        if content.startswith("```"):
            # Remove markdown code block wrapper
            lines = content.split("\n")
            # Find the start and end of JSON
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    start_idx = i + 1
                    break
            
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("```"):
                    end_idx = i
                    break
            
            content = "\n".join(lines[start_idx:end_idx]).strip()

        # Parse JSON safely
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from model:\n{content}")

        return parsed
