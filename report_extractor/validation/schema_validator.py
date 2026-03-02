# midas_extraction/validation/schema_validator.py

from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from datetime import date
from instructor import Instructor  # used later to enforce constraints in prompt


class LabTest(BaseModel):
    test_name: str = Field(..., description="Name of the lab test as it appears on the report (e.g., 'Hemoglobin', 'Glucose', 'White Blood Cell Count'). Look for the test name in the left column or header of the lab results table.")
    value: str = Field(..., description="The measured value returned by the lab (e.g., '13.5', 'Negative', 'Normal'). This is typically a number or categorical result. Keep as string to preserve precision and categorical results.")
    unit: Optional[str] = Field(default=None, description="Unit of measurement following the value (e.g., 'g/dL' for hemoglobin, 'mg/dL' for glucose, 'cells/μL' for blood counts). Usually appears directly after the numeric value.")
    reference_range: Optional[str] = Field(default=None, description="Normal/reference range for this test (e.g., '13.5-17.5 g/dL', '70-100 mg/dL'). Often labeled as 'Reference Range', 'Normal Range', 'Reference Interval', or 'ReferenceRange'. May be in a separate column.")
    abnormal_flag: Optional[bool] = Field(default=None, description="True if the value is flagged as abnormal/critical on the report. Often marked with symbols (*, H for high, L for low, ↑, ↓) or highlighted in color. Set to False if within normal range.")


class PatientInfo(BaseModel):
    name: Optional[str] = Field(default=None, description="Patient full name as it appears on the report (e.g., 'John Smith', 'Maria Garcia'). Usually located at the top of the document in a 'Patient Name' or 'Patient' field. Include first and last name if available.")
    age: Optional[int] = Field(default=None, description="Patient age in years as a numeric value. May be explicitly labeled as 'Age', or can be calculated from date of birth (DOB). Extract only the number of years.")
    gender: Optional[str] = Field(default=None, description="Patient gender typically labeled as 'M' (Male), 'F' (Female), or full words 'Male'/'Female'. Found near patient demographics at the top of the report.")
    report_date: Optional[date] = Field(default=None, description="Date when the lab work was performed and/or the report was issued (e.g., '2024-02-15'). Usually labeled as 'Date of Test', 'Report Date', 'Specimen Date', or 'Date Collected'. Format: YYYY-MM-DD.")


class MedicalReport(BaseModel):
    patient: Optional[PatientInfo] = Field(default=None, description="Patient demographic information extracted from the report header. Contains name, age, gender, and report date.")
    lab_tests: List[LabTest] = Field(default_factory=list, description="List of all lab test results found on the report. Each test includes name, value, unit, reference range, and abnormal flag. Extract all tests from result tables or lists.")
    diagnosis: Optional[List[str]] = Field(default=None, description="Clinical diagnoses, impressions, or conclusions from the report (e.g., ['Anemia', 'Elevated glucose levels']). Usually found in 'Diagnosis', 'Impression', 'Clinical Remarks', or 'Conclusion' sections. Does not include individual lab findings.")
    notes: Optional[str] = Field(default=None, description="Any additional clinical notes, recommendations, or comments from the physician (e.g., 'Follow-up testing recommended in 2 weeks'). Typically found in 'Notes', 'Recommendations', 'Comments', or 'Remarks' sections at the end of the report.")


class SchemaValidator:
    """
    Wraps Pydantic validation for downstream use.
    """

    @staticmethod
    def validate(report_dict: dict) -> MedicalReport:
        try:
            return MedicalReport.model_validate(report_dict)
        except ValidationError as e:
            # Optionally: log or flag errors
            raise ValueError(f"Schema validation failed: {e}")
