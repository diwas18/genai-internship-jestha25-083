import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def clean_text(text: str) -> str:
    """Removes extra spaces and normalizes line breaks."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Splits a long string into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # Shift forward with overlap

    return chunks


def process_document():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)

    # 1. Sample long document
    sample_document = """
    Distributed systems are collections of independent computers that appear to the user as a single coherent system.
    Key challenges in distributed systems include message loss, network partitions, latency, and clock drift.
    
    To maintain order across distributed nodes, algorithms like Lamport Timestamps, Vector Clocks, and mutual exclusion
    algorithms like Ricart-Agrawala are used. Consistency models determine how state updates are propagated across nodes.
    
    Remote Procedure Call (RPC) protocols allow a program to cause a subroutine or procedure to execute in another address space.
    This hides the underlying network communication layers from developers, enabling seamless inter-process communication.
    """

    # 2. Clean and chunk the document
    cleaned_doc = clean_text(sample_document)
    chunks = chunk_text(cleaned_doc, chunk_size=40, overlap=10)

    print(f"Total Chunks Generated: {len(chunks)}\n")

    # 3. Process each chunk with Gemini API
    config = types.GenerateContentConfig(
        system_instruction="You are a precise technical text processor. Summarize the given text chunk in a single concise sentence.",
        temperature=0.2,
    )

    for idx, chunk in enumerate(chunks, start=1):
        print(f"--- Processing Chunk {idx}/{len(chunks)} ---")
        print(f"Chunk Preview: \"{chunk[:60]}...\"")

        success = False
        retries = 3

        for attempt in range(retries):
            time.sleep(3)  # Delay between calls/retries
            try:
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=f"Summarize this text: {chunk}",
                    config=config,
                )
                print(f"Summary: {response.text.strip()}\n")
                success = True
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to process Chunk {idx} after {retries} attempts.\n")


if __name__ == "__main__":
    process_document()