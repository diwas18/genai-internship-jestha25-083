import os
import sys
import numpy as np
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set in .env file.")
    sys.exit(1)

EMBED_MODEL = "gemini-embedding-001"


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates cosine similarity between two numerical vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def chunk_text_by_words(text: str, chunk_size: int = 30, overlap: int = 10) -> list[str]:
    """Splits text into chunks of specified word length with sliding overlap."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def main():
    print("==================================================")
    print("  Gemini API — Week 5 Day 3: Text Chunking Search")
    print("==================================================\n")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Long document sample
    long_document = (
        "Retrieval-Augmented Generation, or RAG, is an architectural pattern that improves the quality "
        "of Large Language Model responses by grounding the model on external resources. Instead of relying "
        "solely on static parameter weights learned during training, RAG fetches relevant context dynamically. "
        "The first phase involves chunking documents into smaller text segments, generating vector embeddings "
        "for each segment, and indexing them in a vector database. When a user sends a query, the system embeds "
        "the query using the same embedding model and executes a cosine similarity search against the vector index. "
        "The most relevant chunks are retrieved and prepended to the user prompt, allowing the LLM to synthesize "
        "an accurate answer based on fresh, custom knowledge."
    )

    print("1. Chunking long document...")
    chunks = chunk_text_by_words(long_document, chunk_size=30, overlap=10)
    for idx, c in enumerate(chunks, start=1):
        print(f"   [Chunk {idx}]: {c}")
    print()

    print("2. Generating batch embeddings for chunks...")
    batch_response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=chunks
    )
    chunk_embeddings = [emb.values for emb in batch_response.embeddings]
    print(f"Generated {len(chunk_embeddings)} chunk vectors.\n")

    # Query targeting a specific section of the document
    query = "How does RAG query the vector index and generate an answer?"
    print(f"3. User Query: '{query}'")

    query_response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query
    )
    query_embedding = query_response.embeddings[0].values

    # Find best matching chunk
    results = []
    for idx, (chunk, chunk_emb) in enumerate(zip(chunks, chunk_embeddings), start=1):
        score = cosine_similarity(query_embedding, chunk_emb)
        results.append((idx, chunk, score))

    results.sort(key=lambda x: x[2], reverse=True)

    print("\n--- Ranked Chunks by Similarity ---")
    for chunk_num, text, score in results:
        print(f"Score: {score:.4f} | Chunk #{chunk_num}: \"{text}\"")
    print("------------------------------------\n")


if __name__ == "__main__":
    main()