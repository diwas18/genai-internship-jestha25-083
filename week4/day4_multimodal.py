import os
import sys
from PIL import Image
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set in .env file.")
    sys.exit(1)

def main():
    print("==================================================")
    print("      Gemini API — Week 4 Day 4: Multimodal")
    print("==================================================\n")

    client = genai.Client()

    # Create a simple test image if no local image is provided
    image_path = "sample_image.png"
    if not os.path.exists(image_path):
        img = Image.new("RGB", (300, 300), color=(73, 109, 137))
        img.save(image_path)
        print(f"Created temporary sample image at '{image_path}'.\n")

    # Load image using Pillow
    pil_image = Image.open(image_path)

    prompt = "Describe what you see in this image in detail, including primary colors, layout, and any notable features."

    print(f"Sending multimodal request with image '{image_path}'...\n")

    try:
        # Pass image object directly in contents list alongside text
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[pil_image, prompt]
        )

        print("--- Output from Gemini ---")
        print(response.text)
        print("---------------------------\n")

    except Exception as e:
        print(f"[Error]: {e}")

if __name__ == "__main__":
    main()