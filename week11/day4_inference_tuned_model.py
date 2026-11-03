import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

print("==================================================")
print("   Week 11 — Day 4: Inference on Fine-Tuned Models")
print("==================================================\n")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# 1. Target endpoint definition (Tuned model resource name vs Base fallback)
TUNED_MODEL_ENDPOINT = "tunedModels/academic-ledger-parser-1042"
FALLBACK_BASE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

system_instruction = "You are an administrative academic log assistant that converts teacher notes into standardized JSON records."
test_prompt = "Aditya Kandel score 4.00 gpa top in class 7 lily."

def run_inference(model_name: str) -> bool:
    print(f"Attempting inference against endpoint [{model_name}]...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=test_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            )
        )
        print("\n--- Successful Model Response ---")
        print(response.text)
        return True
    except Exception as e:
        print(f"[{model_name} Endpoint Failed/Unavailable]: {e}\n")
        return False

# Execution pipeline with fallback routing
print("Initiating deployment endpoint check...\n")
success = run_inference(TUNED_MODEL_ENDPOINT)

if not success:
    print("Tuned model endpoint unavailable. Activating production fallback base model chain...\n")
    for base_model in FALLBACK_BASE_MODELS:
        if run_inference(base_model):
            print(f"Successfully served request via fallback model: [{base_model}]")
            break