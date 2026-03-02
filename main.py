# main.py

import json
import sys
import logging
from pathlib import Path

from report_extractor.extractor import ReportExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

INPUT_DIR = Path("report_extractor/input_reports")
OUTPUT_DIR = Path("report_extractor/output_reports")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def main() -> None:
    extractor = ReportExtractor(output_dir=str(OUTPUT_DIR))

    # If a specific file is passed as argument, process only that file
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        # Process every image in the input_reports folder
        targets = sorted(
            p for p in INPUT_DIR.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    if not targets:
        print(f"No images found in {INPUT_DIR}. Place report images there or pass paths as arguments.")
        sys.exit(1)

    for image_path in targets:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")
        try:
            result = extractor.run(str(image_path))
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            logging.error("Failed to process %s: %s", image_path, e, exc_info=True)


if __name__ == "__main__":
    main()
