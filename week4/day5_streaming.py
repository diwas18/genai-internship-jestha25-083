import os
import sys
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set in .env file.")
    sys.exit(1)

def main():
    print("==================================================")
    print("      Gemini API — Week 4 Day 5: Streaming")
    print("==================================================\n")

    client = genai.Client()

    prompt = "Write a short, engaging story about a developer who discovers their code is running on a quantum computer in another dimension."

    print(f"User Query: {prompt}\n")
    print("--- Streaming Response ---")

    try:
        # Use generate_content_stream to receive output chunks in real time
        response = client.models.generate_content_stream(
            model="gemini-3.6-flash",
            contents=prompt
        )

        for chunk in response:
            # Print each chunk as it arrives without newline buffering
            print(chunk.text, end="", flush=True)

        print("\n---------------------------\n")
        print("Stream completed successfully.")

    except Exception as e:
        print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()