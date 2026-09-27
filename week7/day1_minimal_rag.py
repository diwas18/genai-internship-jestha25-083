import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()

print("==================================================")
print("   Week 7 — Day 1: Minimal RAG Loop")
print("==================================================\n")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-flash-latest"  # alias — won't go stale as models get retired

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client):
        self.client = client

    def __call__(self, input: Documents) -> Embeddings:
        return [
            self.client.models.embed_content(model=EMBED_MODEL, contents=text)
            .embeddings[0].values
            for text in input
        ]


documents = [
    "Distributed systems integrate independent computers into a single coherent system, managing challenges like network partitions and latency.",
    "Algorithms like Lamport Timestamps, Vector Clocks, and Ricart-Agrawala maintain ordering and mutual exclusion across distributed nodes.",
    "Remote Procedure Call (RPC) protocols allow a program to execute code in another address space, hiding underlying network complexities.",
    "Consistency models dictate how state updates are propagated and observed across independent nodes in a system. Strong consistency guarantees every read sees the latest write, while eventual consistency trades that guarantee for higher availability.",
]

chroma_client = chromadb.PersistentClient(path="./chroma_db")
existing = [c.name for c in chroma_client.list_collections()]
if "week7_rag" in existing:
    chroma_client.delete_collection(name="week7_rag")

collection = chroma_client.get_or_create_collection(
    name="week7_rag",
    embedding_function=GeminiEmbeddingFunction(client),
)
ids = [f"doc_{i}" for i in range(len(documents))]
collection.add(documents=documents, ids=ids)


def retrieve(query: str, n_results: int = 2) -> list[str]:
    """Step 1: RETRIEVE relevant chunks from the vector DB."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


def generate_answer(query: str, context_chunks: list[str], max_retries: int = 5) -> str:
    """Step 2: GENERATE an answer using only the retrieved context."""
    context = "\n\n".join(f"- {chunk}" for chunk in context_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt,
            )
            return response.text
        except errors.ClientError as e:
            # 404, bad model name, bad request — retrying won't help, fail fast
            raise RuntimeError(f"Non-retryable error calling '{CHAT_MODEL}': {e}") from e
        except Exception as e:
            # Transient errors (503 overload, timeouts, etc.) — retry with backoff
            if attempt == max_retries:
                raise
            wait = min(2 ** attempt, 30)
            print(f"[Retry {attempt}/{max_retries}] {e}. Waiting {wait}s...")
            time.sleep(wait)


query = "How do RPC protocols simplify writing distributed code?"
print(f"Question: {query}\n")

retrieved_chunks = retrieve(query)
print("--- Retrieved context ---")
for i, chunk in enumerate(retrieved_chunks, start=1):
    print(f"[{i}] {chunk}")
print()

answer = generate_answer(query, retrieved_chunks)
print("--- Generated answer ---")
print(answer)