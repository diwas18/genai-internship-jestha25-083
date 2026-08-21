import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Load environment variables
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

def main():
    print("==================================================")
    print("   Gemini CLI Assistant — Week 3 Final Build")
    print("==================================================")
    print("Commands: 'history' (view memory), 'exit' (quit)\n")

    # Day 2 & Day 4: System prompt + Generation parameters
    config = types.GenerateContentConfig(
        system_instruction="You are a helpful, precise, and practical programming assistant.",
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1000
    )

    try:
        client = genai.Client()
        # Day 3: Multi-turn chat session
        chat = client.chats.create(
            model="gemini-3.6-flash",
            config=config
        )
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nExiting session. Goodbye!")
                break

            # Day 3: Inspect active history
            if user_input.lower() == "history":
                print("\n--- Active Chat History ---")
                for message in chat.get_history():
                    role = "User" if message.role == "user" else "Bot"
                    text = message.parts[0].text if message.parts else ""
                    print(f"[{role}]: {text}")
                print("---------------------------\n")
                continue

            print("\nBot: ", end="", flush=True)

            # Day 1: Streaming response delivery
            response = chat.send_message_stream(user_input)
            for chunk in response:
                print(chunk.text, end="", flush=True)

            print("\n")

        # Day 5: Error handling
        except APIError as e:
            print(f"\n[API Error]: Code {e.code} - {e.message}\n")
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user. Exiting cleanly...")
            break
        except Exception as e:
            print(f"\n[Unexpected Error]: {e}\n")

if __name__ == "__main__":
    main()