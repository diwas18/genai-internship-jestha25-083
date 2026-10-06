import os
import sys
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()

print("==================================================")
print("   Week 9 — Day 5: Synthesis & Architectural Comparison")
print("==================================================\n")

MODELS = [
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


# ---------- Ground Truth & Local Tools ----------

CLASS_LEDGER = {
    "Aditya Kandel": {"gpa": 4.00, "attendance_pct": 98.5},
    "Swikrit Ghimire": {"gpa": 3.85, "attendance_pct": 94.0},
    "Sinchal Rijal": {"gpa": 3.90, "attendance_pct": 96.2},
    "Aarjan Bhusal": {"gpa": 3.60, "attendance_pct": 89.0},
}


def compute_weighted_academic_score(gpa: float, attendance_pct: float) -> dict:
    """Calculates a student's weighted academic composite score.
    Formula: (GPA / 4.0 * 70) + (Attendance_PCT * 0.3)

    Args:
        gpa: Grade Point Average (0.0 to 4.0 scale).
        attendance_pct: Student attendance percentage (0 to 100).
    """
    print(f"   [tool called] compute_weighted_academic_score(gpa={gpa}, attendance_pct={attendance_pct})")
    score = (gpa / 4.0 * 70.0) + (attendance_pct * 0.3)
    return {
        "gpa": gpa,
        "attendance_pct": attendance_pct,
        "composite_score": round(score, 2)
    }


def lookup_student_record(student_name: str) -> dict:
    """Retrieves student GPA and attendance percentage from the class ledger database.

    Args:
        student_name: Full name of the student.
    """
    print(f"   [tool called] lookup_student_record({student_name!r})")
    for name, data in CLASS_LEDGER.items():
        if name.lower() == student_name.strip().lower():
            return {"student": name, **data}
    return {"error": f"Student '{student_name}' not found."}


# Tool configuration for AFC execution
afc_config = types.GenerateContentConfig(
    tools=[lookup_student_record, compute_weighted_academic_score],
    temperature=0.0
)


def run_synthesis_comparison(query: str):
    print("=" * 70)
    print(f"QUERY: \"{query}\"")
    print("=" * 70)

    # --- Approach A: Plain Prompting (No Tools) ---
    print("\n--- Approach A: Plain Prompting (No Tools) ---")
    for model in MODELS:
        try:
            res = client.models.generate_content(
                model=model,
                contents=f"Answer the query based purely on your knowledge. Query: {query}"
            )
            print(f"[{model} Output]:\n{res.text}\n")
            break
        except Exception as e:
            print(f"[{model} Failed]: {e}")

    # --- Approach B: Automatic Function Calling (AFC) ---
    print("--- Approach B: Automatic Function Calling (AFC) ---")
    for model in MODELS:
        try:
            chat = client.chats.create(model=model, config=afc_config)
            res = chat.send_message(query)
            print(f"\n[{model} Final Answer]:\n{res.text}\n")
            break
        except Exception as e:
            print(f"[{model} Failed]: {e}")

    print("=" * 70 + "\n")


# Execution Test Cases
run_synthesis_comparison(
    "Lookup Aditya Kandel's record and calculate his composite academic score."
)