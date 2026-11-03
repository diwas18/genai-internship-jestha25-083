import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

print("==================================================")
print("   Week 10 — Day 1: Standard JSON Output Mode")
print("==================================================\n")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

MODELS = [
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

# Configure generation config to demand application/json
config = types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.1
)

prompt = """
Extract student profile information and return a single JSON object.
Fields required:
- student_id (integer)
- full_name (string)
- gpa (float)
- subjects (list of strings)
- is_eligible_for_honors (boolean)

Text to extract from:
"Student ID 1042 is Aditya Kandel. He achieved a 4.00 GPA with coursework in Computer Science, Mathematics, and Physics. He is eligible for honors."
"""

for model in MODELS:
    try:
        print(f"Requesting JSON output from [{model}]...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        
        # Verify and parse response as valid JSON
        json_data = json.loads(response.text)
        print("\n--- Raw Validated JSON Response ---")
        print(json.dumps(json_data, indent=2))
        print("\nSuccessfully parsed JSON object!")
        break
    except Exception as e:
        print(f"[{model} Failed]: {e}")