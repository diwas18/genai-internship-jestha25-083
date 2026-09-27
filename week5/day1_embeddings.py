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


def main():
    print("==================================================")
    print("    Gemini API — Week 5: Embeddings & Search")
    print("==================================================\n")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Small sample dataset
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps across a sleepy hound.",
        "Python is a popular programming language for data science.",
        "Artificial intelligence and machine learning are transforming tech.",
        "Delicious Nepalese momos are served with spicy tomato chutney."
    ]

    print("1. Generating embeddings for sample dataset...")
    doc_embeddings = []
    for doc in documents:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=doc
        )
        doc_embeddings.append(response.embeddings[0].values)

    print(f"Generated {len(doc_embeddings)} embeddings (Vector size: {len(doc_embeddings[0])})\n")

    # Sample query
    query = "Where can I find tasty dumplings in Nepal?"
    print(f"2. Query: '{query}'")

    query_response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query
    )
    query_embedding = query_response.embeddings[0].values

    # Calculate similarity score for each document
    print("\n3. Calculating similarity scores...")
    results = []
    for i, doc_emb in enumerate(doc_embeddings):
        similarity = cosine_similarity(query_embedding, doc_emb)
        results.append((documents[i], similarity))

    # Sort results by similarity score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Search Results (Ranked by Relevance) ---")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"[{rank}] Score: {score:.4f} | Document: '{doc}'")
    print("--------------------------------------------\n")


if __name__ == "__main__":
    main()