import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

print("==================================================")
print("   Week 10 — Day 4: Extracting Unstructured Data")
print("==================================================\n")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

MODELS = [
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]


# ---------- Data Models for System Log Extraction ----------

class SeverityLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogEntry(BaseModel):
    timestamp: str = Field(description="ISO format timestamp or raw time string extracted from log.")
    severity: SeverityLevel = Field(description="Categorized log severity level.")
    service_name: str = Field(description="System component or service name (e.g., Auth, Database, API).")
    message: str = Field(description="Cleaned main error or log detail message.")
    error_code: Optional[int] = Field(default=None, description="HTTP or internal error code if present.")

class SystemAuditReport(BaseModel):
    environment: str = Field(description="Target environment (e.g., Production, Staging).")
    total_logs_analyzed: int = Field(ge=1, description="Count of log entries parsed.")
    critical_issue_detected: bool = Field(description="True if any ERROR or CRITICAL logs are present.")
    entries: List[LogEntry] = Field(description="Parsed list of individual structured log records.")


config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=SystemAuditReport,
    temperature=0.0
)

# Unstructured messy log transcript
RAW_LOG_DATA = """
[2026-04-12 10:14:02] PROD-ENV - AuthServices: User login successful for ID 4821. (Code 200)
[2026-04-12 10:14:15] PROD-ENV - DatabaseCluster: Connection timeout while querying student_ledger. Retrying... (WARN)
[2026-04-12 10:14:30] PROD-ENV - PaymentGateway: Critical failure: Failed to reach bank API server. HTTP Status 503.
"""

prompt = f"""
Analyze and convert the following raw system logs into a structured audit report:

{RAW_LOG_DATA}
"""

for model in MODELS:
    try:
        print(f"Parsing unstructured log payload with [{model}]...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        print("\n--- Structured System Audit Report ---")
        print(response.text)

        # Validate with Pydantic
        report = SystemAuditReport.model_validate_json(response.text)
        print("\n[Validation Success]: Log batch successfully parsed into database-ready schema!")
        print(f"Environment: {report.environment} | Critical Alert: {report.critical_issue_detected} | Log Count: {len(report.entries)}")
        break
    except Exception as e:
        print(f"[{model} Failed]: {e}\n")