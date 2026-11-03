import os
import time
import logging
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenAI-Backend")

app = FastAPI(
    title="GenAI Capstone Production Backend",
    description="Production-grade API gateway with custom exception middleware, logging, and Gemini integration.",
    version="3.0.0"
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# ---------- Custom Middleware for Request Timing & Logging ----------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f} sec"
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Duration: {process_time:.4f}s")
    return response


# ---------- Global Exception Handlers ----------

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error occurred: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "path": str(request.url)
        },
    )


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
        "day": 3,
        "stage": "Production Middleware & Exception Handling"
    }

@app.post("/api/v1/extract/student", response_model=StudentRecordResponse)
def extract_student_record(payload: ExtractionRequest):
    """Production endpoint with robust logging and Gemini 3.8 Flash extraction."""
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

        parsed_data = StudentRecordResponse.model_validate_json(response.text)
        return parsed_data

    except Exception as e:
        logger.exception("Extraction pipeline encountered an unexpected error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction pipeline error: {str(e)}"
        )