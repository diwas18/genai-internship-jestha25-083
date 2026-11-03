import time
import requests

url = "http://127.0.0.1:8000/api/v1/extract/student"
payload = {
    "raw_text": "Aditya Kandel secured student ID 101 with a GPA of 3.95 and top academic standing."
}

max_retries = 3
backoff_factor = 2

for attempt in range(1, max_retries + 1):
    print(f"Sending POST request to {url} (Attempt {attempt}/{max_retries})...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("\nSuccess!")
        print("Response JSON:")
        print(response.json())
        break
    elif response.status_code == 503 and attempt < max_retries:
        sleep_time = backoff_factor ** attempt
        print(f"Server busy (503). Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        print(f"\nStatus Code: {response.status_code}")
        print("Response JSON:")
        print(response.json())
        break