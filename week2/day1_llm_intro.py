import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()


def test_llm_connection():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found. Check your .env file.")
        return

    client = genai.Client(api_key=api_key)

    print("Sending prompt to Gemini API...")

    # Wait 3 seconds to clear any leftover rate limits from previous runs
    time.sleep(3)

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents="Explain generative AI in simple terms.",
        )

        print("\n--- LLM Response ---")
        print(response.text)

    except Exception as e:
        print("An error occurred while calling the API:", e)


if __name__ == "__main__":
    test_llm_connection()