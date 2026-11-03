import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(
    title="GenAI Academic Ledger Backend",
    description="Production API wrapper for automated student record and log extraction using Gemini.",
    version="1.0.0"
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# ---------- Pydantic Request / Response Models ----------

class ExtractionRequest(BaseModel):
    raw_text: str = Field(description="Raw unstructured text or teacher note to parse.")
    environment: Optional[str] = Field(default="Production", description="Target environment tag.")

class StudentRecordResponse(BaseModel):
    student_name: str
    student_id: int
    gpa: float
    academic_standing: str


# ---------- API Routes ----------

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "GenAI Internship Backend API",
        "batch": "Generative AI Jestha 25 | 083"
    }

@app.post("/extract/student", response_model=StudentRecordResponse)
def extract_student_record(payload: ExtractionRequest):
    """Extract structured student record from unstructured input text."""
    try:
        # Define strict schema for output
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StudentRecordResponse,
            temperature=0.0
        )

        prompt = f"Extract the student record details from this text:\n{payload.raw_text}"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )

        # Validate and return parsed JSON model
        parsed_record = StudentRecordResponse.model_validate_json(response.text)
        return parsed_record

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str_err(e)}")

def str_err(e):
    return str(e)