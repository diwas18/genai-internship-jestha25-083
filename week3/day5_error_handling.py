import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Load environment variables from .env
load_dotenv()

# Verify API Key is present before starting
if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

# Initialize the Gemini API client
client = genai.Client()

def main():
    print("==========================================")
    print("  Error Handling & Resilience (Week 3 - Day 5)")
    print("==========================================")
    print("Type 'exit' or 'quit' to stop.\n")

    config = types.GenerateContentConfig(
        system_instruction="You are a helpful and reliable AI tutor.",
        temperature=0.3
    )

    try:
        chat = client.chats.create(
            model="gemini-3.6-flash",
            config=config
        )
    except Exception as e:
        print(f"Failed to initialize chat session: {e}")
        return

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            print("\nBot: ", end="", flush=True)

            response = chat.send_message_stream(user_input)

            for chunk in response:
                print(chunk.text, end="", flush=True)

            print("\n")

        except APIError as e:
            # Catches Google GenAI API specific exceptions (e.g., rate limits, invalid model names, status errors)
            print(f"\n[API Error]: Code {e.code} - {e.message}\n")
        except KeyboardInterrupt:
            # Handles CTRL+C gracefully
            print("\n\nSession interrupted by user. Exiting...")
            break
        except Exception as e:
            # Catches all other unexpected runtime errors
            print(f"\n[Unexpected Error]: {e}\n")

if __name__ == "__main__":
    main()