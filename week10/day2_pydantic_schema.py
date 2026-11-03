import os
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

print("==================================================")
print("   Week 10 — Day 2: Pydantic Schema Enforcement")
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


# ---------- Define Pydantic Schema ----------

class CourseGrade(BaseModel):
    subject: str = Field(description="Name of the academic course or subject.")
    grade_letter: str = Field(description="Letter grade earned (e.g., A, B+, C).")
    credit_hours: int = Field(description="Credit hours allocated to the course.")

class StudentReportCard(BaseModel):
    student_name: str = Field(description="Full name of the student.")
    student_id: int = Field(description="Unique numeric student identification number.")
    term: str = Field(description="Academic semester or term (e.g., Spring 2026).")
    gpa: float = Field(description="Cumulative grade point average on a 4.0 scale.")
    courses: List[CourseGrade] = Field(description="Detailed breakdown of enrolled courses and grades.")
    academic_standing: str = Field(description="Official academic standing (e.g., Dean's List, Good Standing).")


# Configure SDK with strict response schema
config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=StudentReportCard,
    temperature=0.0
)

prompt = """
Extract the complete student report card from the following record:
Student ID: 1042
Name: Aditya Kandel
Semester: Spring 2026
Overall GPA: 4.00
Academic Status: Dean's List

Courses Taken:
1. Computer Science (4 credits) - Grade A
2. Advanced Mathematics (3 credits) - Grade A
3. Physics (3 credits) - Grade A
"""

for model in MODELS:
    try:
        print(f"Requesting schema-enforced output from [{model}]...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        print("\n--- Raw JSON Conforming to Pydantic Schema ---")
        print(response.text)

        # Validate structured data using Pydantic instantiation
        report = StudentReportCard.model_validate_json(response.text)
        print("\n[Pydantic Validation]: Successfully instantiated StudentReportCard object!")
        print(f"Student: {report.student_name} | Courses Count: {len(report.courses)}")
        break
    except Exception as e:
        print(f"[{model} Failed]: {e}\n")