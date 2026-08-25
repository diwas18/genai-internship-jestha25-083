import os
import sys
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set in .env file.")
    sys.exit(1)

# 1. Define the target JSON Schema using Pydantic
class TaskItem(BaseModel):
    task_name: str = Field(description="Name or title of the task")
    priority: str = Field(description="Priority level: High, Medium, or Low")
    estimated_hours: float = Field(description="Estimated hours to complete")

class TextAnalysisReport(BaseModel):
    summary: str = Field(description="Concise summary of the provided text")
    sentiment: str = Field(description="Overall sentiment: Positive, Neutral, or Negative")
    key_topics: List[str] = Field(description="Top 3-5 main topics mentioned")
    action_items: List[TaskItem] = Field(description="List of extracted actionable items")

def main():
    print("==================================================")
    print("   Gemini API — Week 4 Day 1: Structured Output")
    print("==================================================\n")

    client = genai.Client()

    sample_text = """
    We had a team sync meeting regarding the upcoming app deployment. Overall, progress is great 
    and the team is enthusiastic, but we hit two main blockers: database migration latency and missing unit tests. 
    Sarah needs to optimize the database query indexes by Friday, which should take about 3 hours. 
    Alex needs to write edge-case unit tests for the auth module, taking roughly 5 hours. 
    We decided to move the target deployment date to next Tuesday.
    """

    print("Analyzing input text and generating structured JSON report...\n")

    # 2. Enforce Pydantic Schema via GenerateContentConfig
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TextAnalysisReport,
        temperature=0.1
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=f"Analyze the following meeting notes:\n{sample_text}",
            config=config
        )

        print("--- Raw JSON Output from Gemini ---")
        print(response.text)
        print("------------------------------------\n")

        # 3. Parse and validate raw response back into native Pydantic object
        parsed_report = TextAnalysisReport.model_validate_json(response.text)
        
        print(f"Summary: {parsed_report.summary}")
        print(f"Sentiment: {parsed_report.sentiment}")
        print(f"Topics: {', '.join(parsed_report.key_topics)}")
        print("\nExtracted Action Items:")
        for item in parsed_report.action_items:
            print(f"  - [{item.priority}] {item.task_name} ({item.estimated_hours}h)")

    except Exception as e:
        print(f"[Error]: {e}")

if __name__ == "__main__":
    main()