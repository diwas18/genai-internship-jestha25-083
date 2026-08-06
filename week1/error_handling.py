# week1/error_handling.py
import json

def read_json_safely(filepath: str):
    """Demonstrates error handling for file reading and JSON parsing."""
    print(f"Attempting to read: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            print("Successfully loaded JSON:")
            print(data)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' does not contain valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    print("--- Testing Valid File ---")
    read_json_safely("week1/intern_info.json")

    print("\n--- Testing Missing File ---")
    read_json_safely("week1/missing_file.json")

if __name__ == "__main__":
    main()