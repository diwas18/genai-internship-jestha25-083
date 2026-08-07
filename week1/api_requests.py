import json
import requests


def get_user_data():
    # Public API endpoint that returns sample user profiles
    url = "https://jsonplaceholder.typicode.com/users"

    print("Fetching user data from API...")

    try:
        # Send HTTP GET request with a 5-second timeout
        response = requests.get(url, timeout=5)

        # Print the HTTP status code (200 means success)
        print("Status Code:", response.status_code)

        # Check if the request was successful
        response.raise_for_status()

        # Convert the raw JSON response into a Python list
        users = response.json()
        print(f"Successfully fetched {len(users)} users.\n")

        # Loop through the first 3 users and print key details
        print("--- Preview of First 3 Users ---")
        for user in users[:3]:
            print(f"ID: {user['id']}")
            print(f"Name: {user['name']}")
            print(f"Email: {user['email']}")
            print(f"City: {user['address']['city']}")
            print(f"Company: {user['company']['name']}\n")

        # Prepare a formatted list of dictionaries to save locally
        data_to_save = []
        for user in users:
            data_to_save.append(
                {
                    "id": user["id"],
                    "name": user["name"],
                    "username": user["username"],
                    "email": user["email"],
                    "city": user["address"]["city"],
                    "company": user["company"]["name"],
                }
            )

        # Write the processed data to a local JSON file
        output_file = "week1/api_response.json"
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(data_to_save, file, indent=4)

        print(f"Data successfully processed and saved to '{output_file}'")

    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not connect to the internet or API server is down."
        )
    except requests.exceptions.Timeout:
        print("Error: The request timed out. Try again later.")
    except requests.exceptions.RequestException as e:
        print("An error occurred during the API request:", e)


if __name__ == "__main__":
    get_user_data()