# report_extractor/extractor.py

import json
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from report_extractor.preprocessing.image_preprocessor import ImagePreprocessor
from report_extractor.extraction.vlm_engine import VLMEngine
from report_extractor.validation.schema_validator import MedicalReport, SchemaValidator

logger = logging.getLogger(__name__)


class ReportExtractor:
    """
    End-to-end pipeline: image → preprocess → VLM extraction → validated JSON.

    Usage:
        extractor = ReportExtractor()
        result = extractor.run("path/to/report.jpg")
        print(result)  # validated dict
    """

    def __init__(
        self,
        model_name: str = "thiagomoraes/medgemma-4b-it:Q8_0",
        temperature: float = 0.0,
        max_dimension: int = 2048,
        output_dir: Optional[str] = None,
    ):
        self.preprocessor = ImagePreprocessor(max_dimension=max_dimension)
        self.vlm = VLMEngine(model_name=model_name, temperature=temperature)
        self.validator = SchemaValidator()
        self.output_dir = Path(output_dir) if output_dir else None

    def run(self, image_path: str) -> dict:
        """
        Full pipeline: load image → preprocess → extract → validate → return dict.

        Args:
            image_path: Path to a medical report image (jpg, png, etc.)

        Returns:
            Validated dict matching MedicalReport schema.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("Loading image: %s", image_path)
        image = Image.open(image_path).convert("RGB")

        # --- Step 1: Preprocess ---
        logger.info("Preprocessing image...")
        processed_image, quality_report = self.preprocessor.smart_preprocess(image)
        logger.info(
            "Quality: %s | Sharpness: %.1f | Contrast: %.1f | Needs preprocessing: %s",
            quality_report.overall_quality,
            quality_report.sharpness,
            quality_report.contrast,
            quality_report.needs_preprocessing,
        )

        # --- Step 2: VLM Extraction ---
        logger.info("Extracting structured data via VLM...")
        raw_result = self.vlm.extract(
            image=processed_image,
            schema=MedicalReport,
        )
        logger.info("Raw extraction complete.")

        # --- Step 3: Validate ---
        logger.info("Validating against schema...")
        validated = self.validator.validate(raw_result)
        result_dict = validated.model_dump(mode="json")
        logger.info("Validation passed.")

        # --- Step 4: Optionally save ---
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_file = self.output_dir / f"{image_path.stem}.json"
            out_file.write_text(json.dumps(result_dict, indent=2, ensure_ascii=False))
            logger.info("Saved output to %s", out_file)

        return result_dict
