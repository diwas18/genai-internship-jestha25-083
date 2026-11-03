import os
import json

print("==================================================")
print("   Week 11 — Day 1: Data Formatting & JSONL Prep")
print("==================================================\n")

# Output path for training data
DATASET_PATH = "week11/training_data.jsonl"

# Domain-specific dataset: Converting teacher ledger notes into structured academic logs
training_examples = [
    {
        "messages": [
            {"role": "system", "content": "You are an administrative academic log assistant that converts teacher notes into standardized JSON records."},
            {"role": "user", "content": "Aditya Kandel score 4.00 gpa top in class 7 lily."},
            {"role": "model", "content": json.dumps({"student_name": "Aditya Kandel", "class": "Class 7 Lily", "gpa": 4.00, "rank": "Top"})},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are an administrative academic log assistant that converts teacher notes into standardized JSON records."},
            {"role": "user", "content": "Swikrit Ghimire got 3.85 gpa second position in class 7 lily."},
            {"role": "model", "content": json.dumps({"student_name": "Swikrit Ghimire", "class": "Class 7 Lily", "gpa": 3.85, "rank": "Second"})},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are an administrative academic log assistant that converts teacher notes into standardized JSON records."},
            {"role": "user", "content": "Sinchal Rijal 3.90 gpa third position in class 7 lily."},
            {"role": "model", "content": json.dumps({"student_name": "Sinchal Rijal", "class": "Class 7 Lily", "gpa": 3.90, "rank": "Third"})},
        ]
    },
]

# Write to JSONL format
os.makedirs("week11", exist_ok=True)
with open(DATASET_PATH, "w", encoding="utf-8") as f:
    for item in training_examples:
        f.write(json.dumps(item) + "\n")

print(f"Successfully generated {len(training_examples)} training examples in '{DATASET_PATH}'!")

# Verify file contents
print("\n--- JSONL Content Preview ---")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())