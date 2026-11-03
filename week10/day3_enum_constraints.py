import os
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

print("==================================================")
print("   Week 10 — Day 3: Enum & Constraint Validation")
print("==================================================\n")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

MODELS = [
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]


# ---------- Enums & Constrained Models ----------

class AcademicStandingEnum(str, Enum):
    DEANS_LIST = "Dean's List"
    GOOD_STANDING = "Good Standing"
    PROBATION = "Academic Probation"

class DepartmentEnum(str, Enum):
    CS = "Computer Science"
    MATH = "Mathematics"
    PHYSICS = "Physics"
    ENG = "Engineering"


class CourseGrade(BaseModel):
    subject: DepartmentEnum = Field(description="Department under which the course is offered.")
    grade_letter: str = Field(description="Letter grade (A, B, C, D, F).")
    credit_hours: int = Field(ge=1, le=5, description="Credit hours (must be between 1 and 5).")

class ConstrainedStudentReport(BaseModel):
    student_name: str = Field(min_length=2, max_length=50, description="Full name of the student.")
    student_id: int = Field(ge=1000, le=9999, description="4-digit numeric student ID.")
    gpa: float = Field(ge=0.0, le=4.0, description="Cumulative GPA strictly bounded between 0.0 and 4.0.")
    academic_standing: AcademicStandingEnum = Field(description="Predefined academic standing category.")
    courses: List[CourseGrade] = Field(description="List of validated course records.")


# SDK Config with constrained schema
config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=ConstrainedStudentReport,
    temperature=0.0
)

prompt = """
Extract student report data for student #1042 Aditya Kandel. 
GPA is 4.0. Status is Dean's List.
Enrolled Courses:
- Computer Science (4 credits, Grade A)
- Mathematics (3 credits, Grade A)
- Physics (3 credits, Grade A)
"""

for model in MODELS:
    try:
        print(f"Requesting constrained schema output from [{model}]...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        print("\n--- Constrained JSON Output ---")
        print(response.text)

        # Validate with Pydantic
        report = ConstrainedStudentReport.model_validate_json(response.text)
        print("\n[Pydantic Validation]: Successfully validated Enums and field constraints!")
        print(f"Standing Enum: {report.academic_standing.value} | Department 1: {report.courses[0].subject.value}")
        break
    except Exception as e:
        print(f"[{model} Failed]: {e}\n")