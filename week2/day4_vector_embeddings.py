import math
import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculates the cosine similarity between two numerical vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def get_embedding(client: genai.Client, text: str) -> list[float]:
    """Retrieves text embedding vector using the google-genai SDK."""
    models = ["gemini-embedding-001"]

    for model in models:
        try:
            res = client.models.embed_content(
                model=model,
                contents=text,
            )
            if hasattr(res, "embedding") and res.embedding:
                return res.embedding.values
            elif hasattr(res, "embeddings") and res.embeddings:
                return res.embeddings[0].values
        except Exception as e:
            print(f"[Debug] Model '{model}' failed: {e}")
            continue

    raise RuntimeError("Could not retrieve embedding from any candidate model.")


def run_semantic_search():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)

    chunks = [
        "Distributed systems integrate independent computers into a single coherent system, managing challenges like network partitions and latency.",
        "Algorithms like Lamport Timestamps, Vector Clocks, and Ricart-Agrawala maintain ordering and mutual exclusion across distributed nodes.",
        "Remote Procedure Call (RPC) protocols allow a program to execute code in another address space, hiding underlying network complexities.",
        "Consistency models dictate how state updates are propagated and observed across independent nodes in a system.",
    ]

    print("Generating embeddings for document chunks...")
    chunk_embeddings = []
    for idx, chunk in enumerate(chunks, start=1):
        emb = get_embedding(client, chunk)
        chunk_embeddings.append(emb)
        time.sleep(1)

    print(
        f"\nGenerated {len(chunk_embeddings)} embeddings (Vector size: {len(chunk_embeddings[0])} dimensions)\n"
    )

    user_query = "How do programs call functions on remote servers?"
    print(f"User Query: \"{user_query}\"\n")

    query_embedding = get_embedding(client, user_query)

    scores = []
    for idx, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings), start=1):
        similarity = cosine_similarity(query_embedding, emb)
        scores.append((idx, chunk, similarity))

    scores.sort(key=lambda x: x[2], reverse=True)

    print("--- Semantic Search Results (Ranked by Relevance) ---")
    for rank, (chunk_num, chunk_text, similarity) in enumerate(scores, start=1):
        print(f"\nRank {rank} (Chunk {chunk_num} - Score: {similarity:.4f}):")
        print(f"\"{chunk_text}\"")


if __name__ == "__main__":
    run_semantic_search()