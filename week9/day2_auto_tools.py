import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()

print("==================================================")
print("   Week 9 — Day 2: Automatic Tool Calling")
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


# ---------- Tool 1: calculator ----------

def calculate(expression: str) -> dict:
    """Evaluates an arithmetic expression and returns the exact result.
    Use this for any math involving multiplication, division, or large numbers.

    Args:
        expression: The arithmetic expression, e.g. '1847 * 293'.
    """
    print(f"   [tool called] calculate({expression!r})")
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return {"error": "Only numbers and + - * / ( ) . are allowed."}
    try:
        return {"result": eval(expression, {"__builtins__": {}}, {})}
    except Exception as e:
        return {"error": f"Could not evaluate: {e}"}


# ---------- Tool 2: lookup ----------

PLANET_FACTS = {
    "mercury": {"diameter_km": 4879, "moons": 0, "note": "Closest planet to the Sun"},
    "venus": {"diameter_km": 12104, "moons": 0, "note": "Hottest planet"},
    "earth": {"diameter_km": 12742, "moons": 1, "note": "Only known planet with life"},
    "mars": {"diameter_km": 6779, "moons": 2, "note": "The Red Planet"},
    "jupiter": {"diameter_km": 139820, "moons": 95, "note": "Largest planet"},
    "saturn": {"diameter_km": 116460, "moons": 146, "note": "Famous for its rings"},
    "uranus": {"diameter_km": 50724, "moons": 28, "note": "Rotates on its side"},
    "neptune": {"diameter_km": 49244, "moons": 16, "note": "Strongest winds"},
}


def lookup_planet(name: str) -> dict:
    """Looks up basic facts (diameter in km, number of moons, a short note)
    about a planet in our Solar System.

    Args:
        name: The planet's name, e.g. 'Jupiter'.
    """
    print(f"   [tool called] lookup_planet({name!r})")
    facts = PLANET_FACTS.get(name.strip().lower())
    if facts is None:
        return {"error": f"No data for '{name}'. Known planets: {', '.join(PLANET_FACTS)}."}
    return facts


# Passing plain functions: the SDK builds the tool descriptions and runs the
# call loop automatically.
config = types.GenerateContentConfig(tools=[calculate, lookup_planet])


def ask(question: str):
    print(f"Question: {question}\n")

    for model in MODELS:
        print(f"Using model: {model}")
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=model, contents=question, config=config
                )
                print("\nFinal answer:")
                print(response.text)
                print("\n" + "=" * 60 + "\n")
                return
            except errors.ServerError:
                if attempt == 3:
                    print(f"  {model} is busy, trying the next model...\n")
                else:
                    time.sleep(2 ** attempt)
            except errors.ClientError as e:
                print(f"  {model} rejected the request ({e}). Trying the next model...\n")
                break

    print("All models are busy right now. Wait a few minutes and run again.")
    print("\n" + "=" * 60 + "\n")


# 1. Needs the calculator only
ask("What is 4821 multiplied by 377?")

# 2. Needs the lookup only
ask("How many moons does Mars have?")

# 3. Needs both: two lookups, then a calculation
ask("How many times wider is Jupiter than Earth? Give the ratio to 1 decimal place.")

# 4. Needs neither
ask("What is the capital of Nepal?")