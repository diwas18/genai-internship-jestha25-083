import os
import sys
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()

print("==================================================")
print("   Week 9 — Day 4: Multi-Tool Agent Chaining")
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


# ---------- Database & Tool Definitions ----------

EMPLOYEE_DB = {
    "lead developer": {"name": "Aditya Kandel", "base_salary": 85000, "performance_rating": "A"},
    "senior designer": {"name": "Swikrit Ghimire", "base_salary": 72000, "performance_rating": "B"},
    "qa engineer": {"name": "Sinchal Rijal", "base_salary": 65000, "performance_rating": "A"},
}

BONUS_RATES = {
    "A": 0.15,  # 15% bonus
    "B": 0.10,  # 10% bonus
    "C": 0.05,  # 5% bonus
}


def get_employee_by_role(role: str) -> dict:
    """Finds employee record (name, base salary, performance rating) by role job title.

    Args:
        role: The job title, e.g., 'Lead Developer' or 'QA Engineer'.
    """
    print(f"   [tool called] get_employee_by_role({role!r})")
    info = EMPLOYEE_DB.get(role.strip().lower())
    if not info:
        return {"error": f"Role '{role}' not found. Available roles: {', '.join(EMPLOYEE_DB.keys())}"}
    return info


def calculate_performance_bonus(base_salary: float, performance_rating: str) -> dict:
    """Calculates bonus amount based on base salary and performance rating.

    Args:
        base_salary: Base salary numerical value.
        performance_rating: Rating grade ('A', 'B', or 'C').
    """
    print(f"   [tool called] calculate_performance_bonus({base_salary!r}, {performance_rating!r})")
    rate = BONUS_RATES.get(performance_rating.upper())
    if rate is None:
        return {"error": f"Invalid rating '{performance_rating}'."}
    bonus = base_salary * rate
    return {"base_salary": base_salary, "bonus_amount": bonus, "total_compensation": base_salary + bonus}


config = types.GenerateContentConfig(
    tools=[get_employee_by_role, calculate_performance_bonus],
    temperature=0.0
)


def ask_agent(query: str):
    print(f"User Query: \"{query}\"\n")

    for model in MODELS:
        print(f"Using model: {model}")
        for attempt in range(1, 4):
            try:
                # client.chats maintains multi-turn function call loops cleanly
                chat = client.chats.create(model=model, config=config)
                response = chat.send_message(query)

                print("\nFinal Answer:")
                print(response.text)
                print("\n" + "=" * 60 + "\n")
                return
            except errors.ServerError:
                if attempt == 3:
                    print(f"  {model} server error. Trying next model...\n")
                else:
                    time.sleep(2 ** attempt)
            except errors.ClientError as e:
                print(f"  {model} client error: {e}. Trying next model...\n")
                break

    print("All models busy or rate limited.")
    print("\n" + "=" * 60 + "\n")


# Multi-step query requiring tool chaining:
ask_agent("Find who the Lead Developer is, and calculate their total compensation including bonus.")