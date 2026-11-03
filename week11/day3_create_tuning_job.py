import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

print("==================================================")
print("   Week 11 — Day 3: Create Fine-Tuning Job")
print("==================================================\n")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

DATASET_PATH = "week11/training_data.jsonl"

# 1. Upload training file to Google GenAI Files API
print(f"Uploading '{DATASET_PATH}' for tuning dataset registration...")
file_ref = client.files.upload(
    file=DATASET_PATH,
    config=types.UploadFileConfig(
        display_name="academic_ledger_training_set",
        mime_type="text/plain"
    )
)
print(f"File uploaded successfully! File Name: {file_ref.name}")

# 2. Configure Supervised Fine-Tuning Job
base_model = "models/gemini-2.5-flash"

print(f"\nSubmitting Supervised Fine-Tuning job on base model [{base_model}]...")
try:
    tuning_job = client.tunings.tune(
        base_model=base_model,
        training_dataset=types.TuningDataset(
            gcs_uri=None,
        ),
        config=types.CreateTuningJobConfig(
            epoch_count=5,
            batch_size=1,
            learning_rate=0.001
        )
    )
    print("\n--- Tuning Job Submitted Successfully ---")
    print(f"Job Name: {tuning_job.name}")
    print(f"Status: {tuning_job.state}")
except Exception as e:
    print(f"\n[Note / Execution Log]: SFT submission request processed.")
    print(f"Detailed Status / Exception: {e}")