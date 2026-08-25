import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set in .env file.")
    sys.exit(1)

# 1. Define python tools/functions that Gemini can invoke
def get_current_weather(location: str) -> str:
    """Get the current weather for a given city location."""
    # Mock response for demonstration
    weather_data = {
        "kathmandu": "18°C, Partly Cloudy",
        "tokyo": "12°C, Sunny",
        "new york": "5°C, Rain"
    }
    loc_lower = location.lower()
    for city, status in weather_data.items():
        if city in loc_lower:
            return f"Weather in {location}: {status}"
    return f"Weather data for {location} is currently unavailable."

def calculate_discount(original_price: float, discount_percentage: float) -> str:
    """Calculates the final price after applying a percentage discount."""
    discounted_amount = original_price * (discount_percentage / 100.0)
    final_price = original_price - discounted_amount
    return f"Original: ${original_price:.2f}, Discount: {discount_percentage}%, Final Price: ${final_price:.2f}"

def main():
    print("==================================================")
    print("   Gemini API — Week 4 Day 2: Function Calling")
    print("==================================================\n")

    client = genai.Client()

    # Map of callable functions for local execution
    tools_map = {
        "get_current_weather": get_current_weather,
        "calculate_discount": calculate_discount
    }

    prompt = "What's the weather like in Kathmandu right now, and how much would a $120 jacket cost with a 20% discount?"
    print(f"User Query: {prompt}\n")

    # 2. Register tools with GenerateContentConfig
    config = types.GenerateContentConfig(
        tools=[get_current_weather, calculate_discount],
        temperature=0.1
    )

    try:
        # Step A: Send request to Gemini with registered functions
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=config
        )

        # Check if Gemini requested function calls
        if response.function_calls:
            print("--- Function Calls Triggered by Gemini ---")
            for call in response.function_calls:
                fn_name = call.name
                fn_args = call.args
                print(f"Tool Requested: {fn_name}({fn_args})")
                
                # Execute local Python function
                if fn_name in tools_map:
                    result = tools_map[fn_name](**fn_args)
                    print(f"Tool Execution Result: {result}\n")
        else:
            print("No function calls triggered.")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"[Error]: {e}")

if __name__ == "__main__":
    main()