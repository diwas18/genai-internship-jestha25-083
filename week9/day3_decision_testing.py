import os
import sys
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()

print("==================================================")
print("   Week 9 — Day 3: Tool Decision Testing")
print("==================================================\n")

MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
]

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


# ---------- Tool 1: Math Operation ----------

def calculate_square_root(number: float) -> dict:
    """Calculates the square root of a non-negative number.

    Args:
        number: The numeric value to calculate the square root for.
    """
    print(f"   [tool called] calculate_square_root({number!r})")
    if number < 0:
        return {"error": "Cannot calculate the square root of a negative number in real numbers."}
    return {"result": number ** 0.5}


# ---------- Tool 2: Student Record Lookup ----------

STUDENT_RECORDS = {
    "Aditya Kandel": "4.00",
    "Swikrit Ghimire": "3.85",
    "Sinchal Rijal": "3.90",
    "Diwas Sigdel": "4.00"
}

def lookup_student_gpa(student_name: str) -> dict:
    """Looks up the academic GPA of a student by their full name.

    Args:
        student_name: The full name of the student.
    """
    print(f"   [tool called] lookup_student_gpa({student_name!r})")
    # Case-insensitive lookup
    for name, gpa in STUDENT_RECORDS.items():
        if name.lower() == student_name.strip().lower():
            return {"student": name, "gpa": gpa}
            
    return {"error": f"Student '{student_name}' was not found in the academic record database."}


# Tool configuration for AFC
config = types.GenerateContentConfig(
    tools=[calculate_square_root, lookup_student_gpa],
    temperature=0.0  # Zero temperature for deterministic tool routing decisions
)


def run_test_case(query: str, test_description: str):
    print(f"--------------------------------------------------")
    print(f"Test: {test_description}")
    print(f"Query: \"{query}\"\n")

    for model in MODELS:
        print(f"Using model: {model}")
        for attempt in range(1, 4):
            try:
                # Chat session ensures AFC retains conversation turns and context
                chat = client.chats.create(model=model, config=config)
                response = chat.send_message(query)
                
                print("\nFinal Answer:")
                print(response.text)
                print("\n" + "=" * 60 + "\n")
                return
            except errors.ServerError:
                if attempt == 3:
                    print(f"  {model} is busy, trying next model...\n")
                else:
                    time.sleep(2 ** attempt)
            except errors.ClientError as e:
                print(f"  {model} request error ({e}). Trying next model...\n")
                break

    print("All models failed or are busy right now.")
    print("\n" + "=" * 60 + "\n")


# 1. Unneeded Tool / General Knowledge (Should answer directly without tool execution)
run_test_case(
    query="What is the chemical symbol for Gold and why is it used in electronics?",
    test_description="General Knowledge (Model should answer directly without invoking tools)"
)

# 2. Tool Execution Required (Valid lookup)
run_test_case(
    query="What is the GPA of Aditya Kandel?",
    test_description="Valid Tool Execution (Model should invoke lookup_student_gpa)"
)

# 3. Tool Error Handling (Missing Record)
run_test_case(
    query="Can you check the GPA for John Doe?",
    test_description="Tool Error / Missing Record (Tool returns error dict, model interprets result cleanly)"
)

# 4. Math Edge Case (Invalid Input)
run_test_case(
    query="What is the square root of -16?",
    test_description="Math Edge Case (Tool returns real-number error, model reports limitation)"
)