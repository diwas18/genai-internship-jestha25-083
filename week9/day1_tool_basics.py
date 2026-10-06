import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()

print("==================================================")
print("   Week 9 — Day 1: Tool Use, Step by Step")
print("==================================================\n")

# Tried in order. If one is overloaded, the script moves to the next.
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


# ---------- 1. The tool itself: a plain Python function ----------

def calculate(expression: str) -> dict:
    """Safely evaluates a basic arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return {"error": "Only numbers and + - * / ( ) . are allowed."}
    try:
        return {"result": eval(expression, {"__builtins__": {}}, {})}
    except Exception as e:
        return {"error": f"Could not evaluate: {e}"}


# ---------- 2. Describe the tool to the model ----------

calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="calculate",
            description=(
                "Evaluates an arithmetic expression and returns the exact result. "
                "Use this for any math involving multiplication, division, or large numbers."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="The arithmetic expression, e.g. '1847 * 293'",
                    )
                },
                required=["expression"],
            ),
        )
    ]
)

# Automatic calling is OFF so we can see and run each step ourselves
config = types.GenerateContentConfig(
    tools=[calculator_tool],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def call_model(model: str, contents, max_retries: int = 3):
    """Calls one model, retrying briefly on overload (503).
    If it is still overloaded, the ServerError is raised so the caller can
    move on to the next model."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except errors.ServerError as e:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"  [{model}] overloaded, retry {attempt}/{max_retries} in {wait}s...")
            time.sleep(wait)


def run_flow(model: str, question: str):
    """The full tool-use flow, using one model from start to finish."""
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    # ---------- 3. First call: does the model want a tool? ----------
    response = call_model(model, contents)

    if not response.function_calls:
        print("Model answered directly (no tool needed):")
        print(response.text)
        return

    # ---------- 4. Model asked for a tool: run it ourselves ----------
    print("Model requested tool call(s):")
    response_parts = []
    for call in response.function_calls:
        print(f"  -> {call.name}({dict(call.args)})")
        result = calculate(**call.args)
        print(f"  <- our function returned: {result}")
        response_parts.append(
            types.Part.from_function_response(name=call.name, response=result)
        )
    print()

    # ---------- 5. Send the result back so the model can answer ----------
    contents.append(response.candidates[0].content)                    # the model's tool request
    contents.append(types.Content(role="user", parts=response_parts))  # our results

    final = call_model(model, contents)
    print("Final answer:")
    print(final.text)


def ask(question: str):
    print(f"Question: {question}\n")

    for model in MODELS:
        print(f"Using model: {model}")
        try:
            run_flow(model, question)
            print("\n" + "=" * 60 + "\n")
            return
        except errors.ServerError:
            print(f"  {model} is busy, trying the next model...\n")
        except errors.ClientError as e:
            print(f"  {model} rejected the request ({e}). Trying the next model...\n")

    print("All models are busy right now. Wait a few minutes and run again.")
    print("\n" + "=" * 60 + "\n")


# Needs the tool
ask("What is 1847 multiplied by 293?")

# Doesn't need the tool
ask("What is the capital of Nepal?")