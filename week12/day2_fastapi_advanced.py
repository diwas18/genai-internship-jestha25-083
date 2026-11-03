import os
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(
    title="GenAI Capstone Production Backend",
    description="Unified API gateway for academic data extraction and system audit logging.",
    version="2.0.1"
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# ---------- Pydantic Schemas ----------

class ExtractionRequest(BaseModel):
    raw_text: str = Field(min_length=5, description="Raw unstructured text input to process.")

class StudentRecordResponse(BaseModel):
    student_name: str
    student_id: int
    gpa: float = Field(ge=0.0, le=4.0)
    academic_standing: str


# ---------- API Endpoints ----------

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "internship": "Generative AI Jestha 25 | 083",
        "week": 12,
        "day": 2,
        "stage": "Advanced Pydantic Integration (Updated Model)"
    }

@app.post("/api/v1/extract/student", response_model=StudentRecordResponse)
def extract_student_record(payload: ExtractionRequest):
    """Production endpoint enforcing strict Pydantic schemas using Gemini 3.8 Flash."""
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StudentRecordResponse,
            temperature=0.0
        )

        prompt = f"Extract the student record details from the following text:\n{payload.raw_text}"

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=config
        )

        # Validate and return parsed JSON
        parsed_data = StudentRecordResponse.model_validate_json(response.text)
        return parsed_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction pipeline error: {str(e)}"
        )