# midas_report_extraction/schemas/medical_schema.py

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class LabTest(BaseModel):
    test_name: str = Field(
        ...,
        description="Name of the laboratory test"
    )

    value: str = Field(
        ...,
        description="Reported test value as written in report"
    )

    unit: Optional[str] = Field(
        None,
        description="Measurement unit if provided"
    )

    reference_range: Optional[str] = Field(
        None,
        description="Reference range if mentioned"
    )

    abnormal_flag: Optional[bool] = Field(
        None,
        description="True if test is marked abnormal"
    )


class PatientInfo(BaseModel):
    name: Optional[str] = Field(
        None,
        description="Patient full name"
    )

    age: Optional[int] = Field(
        None,
        description="Patient age in years"
    )

    gender: Optional[str] = Field(
        None,
        description="Patient gender"
    )

    report_date: Optional[date] = Field(
        None,
        description="Date when report was issued"
    )


class MedicalReport(BaseModel):
    patient: Optional[PatientInfo] = None

    lab_tests: List[LabTest] = Field(
        default_factory=list,
        description="List of laboratory test results"
    )

    diagnosis: Optional[List[str]] = Field(
        None,
        description="List of diagnoses mentioned"
    )

    notes: Optional[str] = Field(
        None,
        description="Additional observations or remarks"
    )
