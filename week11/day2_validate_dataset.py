import json
import os

print("==================================================")
print("   Week 11 — Day 2: Dataset Validation & Schema Check")
print("==================================================\n")

DATASET_PATH = "week11/training_data.jsonl"

def validate_jsonl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    valid_roles = {"system", "user", "model"}
    line_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            # 1. Verify valid JSON syntax
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: Invalid JSON syntax - {e}")

            # 2. Verify top-level key
            if "messages" not in data:
                raise KeyError(f"Line {line_num}: Missing 'messages' list key")

            messages = data["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Line {line_num}: 'messages' must be a list with at least user and model entries")

            # 3. Verify roles and content keys
            for msg in messages:
                if "role" not in msg or "content" not in msg:
                    raise KeyError(f"Line {line_num}: Message entry missing 'role' or 'content'")
                if msg["role"] not in valid_roles:
                    raise ValueError(f"Line {line_num}: Invalid role '{msg['role']}'. Expected one of {valid_roles}")

            line_count += 1

    print(f"[Validation Passed]: All {line_count} dataset records are syntactically valid!")

validate_jsonl(DATASET_PATH)