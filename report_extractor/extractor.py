from PIL import Image
from preprocessing.image_preprocessor import ImagePreprocessor
from extraction.vlm_engine import VLMEngine
from validation.schema_validator import SchemaValidator, MedicalReport

# 1️⃣ Load & preprocess image
img = Image.open("input_reports/report.jpg").convert("RGB")
preprocessor = ImagePreprocessor()
clean_img = preprocessor.preprocess(img)

# 2️⃣ Extract via VLM with MedicalReport schema
vlm = VLMEngine()
raw_json = vlm.extract(clean_img, MedicalReport)

# 3️⃣ Validate & parse JSON
validated_report = SchemaValidator.validate(raw_json)

print(validated_report.model_dump())