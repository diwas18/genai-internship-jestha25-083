import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env
load_dotenv()

# Initialize the Gemini API client
client = genai.Client()

def main():
    print("==========================================")
    print("   System Prompt & Persona (Week 3 - Day 2)")
    print("==========================================")
    print("Type 'exit' or 'quit' to stop.\n")

    # Define system instructions / chatbot persona
    system_instruction = (
        "You are a helpful and concise AI programming tutor. "
        "Keep your explanations clear, beginner-friendly, and brief."
    )

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            print("\nBot: ", end="", flush=True)

            # Pass system_instruction via GenerateContentConfig
            response = client.models.generate_content_stream(
                model="gemini-3.6-flash",
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )

            for chunk in response:
                print(chunk.text, end="", flush=True)

            print("\n")

        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()