import os
import sys
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Initialize the Gemini API client
client = genai.Client()

def main():
    print("==========================================")
    print("      Basic CLI Chatbot (Week 3 - Day 1)  ")
    print("==========================================")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            print("\nBot: ", end="", flush=True)

            # Stream response chunks in real-time
            response = client.models.generate_content_stream(
                model="gemini-3.6-flash",
                contents=user_input
            )

            for chunk in response:
                print(chunk.text, end="", flush=True)

            print("\n")

        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()