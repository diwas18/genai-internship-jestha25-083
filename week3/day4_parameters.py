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
    print("  Generation Parameters Config (Week 3 - Day 4)")
    print("==========================================")
    print("Type 'exit' or 'quit' to stop.\n")

    # Define generation parameters via GenerateContentConfig
    config = types.GenerateContentConfig(
        system_instruction="You are a helpful and precise programming tutor.",
        temperature=0.2,       # Lower value = more deterministic/focused (0.0 to 2.0)
        top_p=0.95,            # Nucleus sampling threshold
        top_k=40,              # Limits candidate token pool
        max_output_tokens=500  # Limits maximum response length
    )

    # Initialize chat session with custom configuration
    chat = client.chats.create(
        model="gemini-3.6-flash",
        config=config
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

            response = chat.send_message_stream(user_input)

            for chunk in response:
                print(chunk.text, end="", flush=True)

            print("\n")

        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
    