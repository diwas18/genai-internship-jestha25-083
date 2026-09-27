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


def chunk_text(text: str, chunk_size: int = 25, overlap: int = 5) -> list[str]:
    """Splits raw text into uniform chunks with overlapping word boundaries."""
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
    print("  Gemini API — Week 5 Deliverable: RAG Pipeline")
    print("==================================================\n")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    raw_knowledge_base = [
        "Retrieval-Augmented Generation (RAG) combines external data retrieval with generative language models.",
        "Vector embeddings store dense mathematical representations of text where semantically related ideas stay close together.",
        "Cosine similarity measures the angle between two embedding vectors in high-dimensional space.",
        "FastAPI is commonly paired with vector stores to create high-throughput semantic search backends.",
        "Chunking prevents long text from diluting embedding signal by isolating focused context windows."
    ]

    # Step 1: Chunking
    print("1. Splitting document corpus into chunks...")
    corpus_chunks = []
    for doc in raw_knowledge_base:
        corpus_chunks.extend(chunk_text(doc, chunk_size=20, overlap=5))
    print(f"   Created {len(corpus_chunks)} chunks.")

    # Step 2: Batch Vectorization
    print("2. Generating batch embeddings for vector index...")
    batch_resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=corpus_chunks
    )
    corpus_vectors = [emb.values for emb in batch_resp.embeddings]
    print(f"   Vectorized {len(corpus_vectors)} chunks.")

    # Step 3: Query & Retrieval
    user_query = "How do vector embeddings measure semantic proximity?"
    print(f"\n3. Query: '{user_query}'")

    query_resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=user_query
    )
    query_vector = query_resp.embeddings[0].values

    # Step 4: Search & Thresholding
    threshold = 0.55
    retrieved_context = []
    
    for chunk, vec in zip(corpus_chunks, corpus_vectors):
        similarity = cosine_similarity(query_vector, vec)
        if similarity >= threshold:
            retrieved_context.append((chunk, similarity))

    retrieved_context.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- Retrieved Context (Threshold >= {threshold}) ---")
    if retrieved_context:
        for rank, (text, score) in enumerate(retrieved_context, start=1):
            print(f"[{rank}] (Score: {score:.4f}) -> \"{text}\"")
    else:
        print("No matching context passed the similarity threshold.")
    print("----------------------------------------------------\n")


if __name__ == "__main__":
    main()