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

def main():
    print("==================================================")
    print("  Gemini API — Week 4 Day 3: Advanced Prompting")
    print("==================================================\n")

    client = genai.Client()

    # 1. System Instruction: Define persona and operating rules
    system_instruction = """
    You are an expert Senior Security & Systems Engineer. 
    Analyze complex technical scenarios by evaluating constraints step-by-step before declaring conclusions.
    Structure your responses using clear headers: [REASONING], [DIAGNOSIS], and [RECOMMENDED ACTION].
    """

    # 2. Few-Shot Examples + Chain of Thought prompt configuration
    prompt = """
    === EXAMPLES ===
    Input: "Our primary database instance is experiencing 99% CPU utilization during peak traffic hours, causing API timeouts."
    Output:
    [REASONING]
    1. Identify high-resource processes: Database execution metrics indicate heavy CPU contention.
    2. Check read vs. write ratio: High read traffic often lacks caching layers or indexed query support.
    3. Evaluate scaling choices: Vertical scaling provides immediate relief; read-replicas distribute query load long-term.
    [DIAGNOSIS] Unindexed database queries combined with elevated read traffic during peak periods.
    [RECOMMENDED ACTION] Implement Redis query caching and create indexes for top slow queries.

    === TASK ===
    Input: "Users are reporting intermittently failing requests to our microservices API gateway with 504 Gateway Timeout errors during high traffic bursts."
    Output:
    """

    print("Sending prompt with System Instructions and Few-Shot CoT structure...\n")

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=config
        )

        print("--- Output from Gemini ---")
        print(response.text)
        print("---------------------------\n")

    except Exception as e:
        print(f"[Error]: {e}")

if __name__ == "__main__":
    main()