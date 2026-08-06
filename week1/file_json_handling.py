# week1/file_json_handling.py
import json

def process_text_file(filepath: str):
    """Demonstrates file creation, string methods, writing, and reading."""
    print("--- 1. File Writing & String Methods ---")
    raw_text = "   Generative AI Internship, Week 1 Basics!   "
    cleaned_text = raw_text.strip().lower()

    # Writing to a text file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"Cleaned text written to '{filepath}'")

    # Reading from the text file
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"Read content: '{content}'\n")

def process_json_file(json_path: str):
    """Demonstrates JSON serialization and deserialization."""
    print("--- 2. JSON File Handling ---")
    data = {
        "intern_name": "Diwas",
        "week": 1,
        "topics_covered": ["String Methods", "File I/O", "Error Handling", "JSON"],
        "status": "Completed"
    }

    # Writing JSON data
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"JSON data successfully written to '{json_path}'")

    # Reading JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    print(f"Loaded JSON Object: {loaded_data}")
    print(f"Intern Name: {loaded_data.get('intern_name')}\n")

def main():
    txt_filename = "week1/sample.txt"
    json_filename = "week1/intern_info.json"

    process_text_file(txt_filename)
    process_json_file(json_filename)

if __name__ == "__main__":
    main()