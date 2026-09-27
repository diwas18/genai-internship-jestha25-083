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
    print("  Gemini API — Week 5 Day 4: Interactive Search")
    print("==================================================\n")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    raw_knowledge_base = [
        "Python is an interpreted, high-level, general-purpose programming language widely used in data science.",
        "Generative AI models use deep neural networks to generate text, images, and audio from user prompts.",
        "Retrieval-Augmented Generation (RAG) connects an LLM to external document stores via vector search.",
        "Vector databases index numerical embeddings using metrics like cosine similarity or Euclidean distance.",
        "FastAPI and Flask are popular Python micro-frameworks for building RESTful Web APIs."
    ]

    # Process documents into chunks
    all_chunks = []
    for doc in raw_knowledge_base:
        all_chunks.extend(chunk_text(doc, chunk_size=20, overlap=5))

    print(f"1. Preprocessing knowledge base into {len(all_chunks)} chunks...")
    
    # Generate batch embeddings for preprocessed chunks
    batch_resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=all_chunks
    )
    chunk_vectors = [emb.values for emb in batch_resp.embeddings]
    print("2. Knowledge base successfully vectorized and indexed in memory.\n")

    print("--- Interactive Semantic Search CLI ---")
    print("Type your query and press Enter (or type 'exit' to quit).\n")

    while True:
        try:
            query = input("Search > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting search session.")
                break

            # Vectorize user query
            query_resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=query
            )
            query_vector = query_resp.embeddings[0].values

            # Perform similarity search against indexed chunks
            matches = []
            for chunk_text_str, chunk_vec in zip(all_chunks, chunk_vectors):
                score = cosine_similarity(query_vector, chunk_vec)
                matches.append((chunk_text_str, score))

            matches.sort(key=lambda x: x[1], reverse=True)

            # Display top 2 results
            print(f"\nTop Matches for '{query}':")
            for rank, (text, score) in enumerate(matches[:2], start=1):
                print(f"  [{rank}] Score: {score:.4f} | \"{text}\"")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nSession terminated.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")


if __name__ == "__main__":
    main()