import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def test_prompt_engineering():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found. Check your .env file.")
        return

    client = genai.Client(api_key=api_key)

    # Config with System Instruction & Parameters
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a concise Senior Python Developer. "
            "Provide brief, clean, and direct technical answers."
        ),
        temperature=0.2,  # Low = factual/deterministic
        max_output_tokens=1000,  # Limits response length
    )

    print("Sending prompt with System Instruction & Temperature=0.2...\n")
    time.sleep(3)

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents="Explain the difference between a list and a tuple in Python.",
            config=config,
        )

        print("--- LLM Response ---")
        print(response.text)

    except Exception as e:
        print("An error occurred while calling the API:", e)


if __name__ == "__main__":
    test_prompt_engineering()