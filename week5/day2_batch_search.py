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
    print("   Gemini API — Week 5 Day 2: Batch Vector Search")
    print("==================================================\n")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Expanded knowledge base
    knowledge_base = [
        "Git is a distributed version control system used for tracking software changes.",
        "Docker allows developers to package applications into lightweight containers.",
        "FastAPI is a modern, fast web framework for building APIs with Python.",
        "PostgreSQL is an advanced, open-source relational database system.",
        "Retrieval-Augmented Generation (RAG) enhances LLMs with custom external context.",
        "Vector databases store high-dimensional embeddings for fast semantic similarity search."
    ]

    print("1. Generating batch embeddings for knowledge base...")
    
    # Pass the list directly to process embeddings in a single batch call
    batch_response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=knowledge_base
    )

    # Extract vectors directly from response embeddings list
    doc_embeddings = [emb.values for emb in batch_response.embeddings]
    print(f"Processed batch of {len(doc_embeddings)} documents. (Vector size: {len(doc_embeddings[0])})\n")

    # Sample search query
    query = "How do vector stores and RAG work with large language models?"
    print(f"2. User Query: '{query}'")

    query_response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query
    )
    query_embedding = query_response.embeddings[0].values

    # Filter results by relevance threshold
    similarity_threshold = 0.60
    print(f"\n3. Calculating similarity scores (Threshold >= {similarity_threshold})...")
    
    results = []
    for doc, doc_emb in zip(knowledge_base, doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        if score >= similarity_threshold:
            results.append((doc, score))

    # Sort results by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Relevant Matches ---")
    if results:
        for rank, (doc, score) in enumerate(results, start=1):
            print(f"[{rank}] Score: {score:.4f} | Document: '{doc}'")
    else:
        print("No documents met the similarity threshold.")
    print("------------------------\n")


if __name__ == "__main__":
    main()