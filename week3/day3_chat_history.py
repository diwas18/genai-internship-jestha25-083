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
    print("   Multi-Turn Chat History (Week 3 - Day 3)")
    print("==========================================")
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'history' to view saved chat turns.\n")

    system_instruction = (
        "You are a helpful and concise AI tutor. "
        "Remember context from previous messages in our conversation."
    )

    # Initialize a multi-turn chat session with system instruction
    chat = client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    )

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            # Custom command to inspect stored history
            if user_input.lower() == "history":
                print("\n--- Conversation History ---")
                for message in chat.get_history():
                    role = "You" if message.role == "user" else "Bot"
                    # Handle multiple parts if present
                    text = "".join([part.text for part in message.parts if part.text])
                    print(f"{role}: {text}")
                print("----------------------------\n")
                continue

            print("\nBot: ", end="", flush=True)

            # Send message and stream the response using stored history context
            response = chat.send_message_stream(user_input)

            for chunk in response:
                print(chunk.text, end="", flush=True)

            print("\n")

        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()