import os
import json
import time
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

print("==================================================")
print("   Week 10 — Day 5: Architectural Benchmark Comparison")
print("==================================================\n")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# Working fallback chain for high-demand scenarios
MODELS = [
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]


# ---------- Data Schema ----------

class StudentLedgerRecord(BaseModel):
    student_name: str = Field(description="Full name of student.")
    gpa: float = Field(ge=0.0, le=4.0, description="GPA score.")
    has_perfect_score: bool = Field(description="True if GPA is exactly 4.00.")

class ClassLedger(BaseModel):
    records: List[StudentLedgerRecord] = Field(description="List of student records.")


RAW_LEDGER_TEXT = """
Class 7 Lily Examination Summary:
1. Aditya Kandel - GPA 4.00 (Perfect GPA)
2. Swikrit Ghimire - GPA 3.85
3. Sinchal Rijal - GPA 3.90
4. Aditya Kandel - GPA 4.00 (Verified second subject distinction)
"""

prompt = f"Parse the following student ledger records:\n{RAW_LEDGER_TEXT}"

# ---------- Approach A: Plain Prompting (Request JSON in string) ----------

def run_approach_a():
    print("--- Approach A: Free-form Plain Text Prompting ---")
    plain_prompt = f"{prompt}\nReturn your result as JSON."
    
    for model in MODELS:
        try:
            res = client.models.generate_content(model=model, contents=plain_prompt)
            print(f"[{model} Output]:")
            print(res.text[:300] + "..." if len(res.text) > 300 else res.text)
            
            # Attempt manual JSON parse
            parsed = json.loads(res.text.strip().strip("```json").strip("```"))
            print("  [Result]: Manual JSON parsing succeeded!\n")
            return
        except Exception as e:
            print(f"  [{model} Failed]: {e}\n")


# ---------- Approach B: Strict Pydantic Schema Enforcement ----------

def run_approach_b():
    print("--- Approach B: Strict Schema-Driven Generation (response_schema) ---")
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ClassLedger,
        temperature=0.0
    )
    
    for model in MODELS:
        try:
            res = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            print(f"[{model} Direct Output]:")
            print(res.text)
            
            # Direct Pydantic validation
            ledger = ClassLedger.model_validate_json(res.text)
            print(f"  [Result]: Pydantic validated {len(ledger.records)} records deterministically!\n")
            return
        except Exception as e:
            print(f"  [{model} Failed]: {e}\n")


# Run comparison
run_approach_a()
run_approach_b()