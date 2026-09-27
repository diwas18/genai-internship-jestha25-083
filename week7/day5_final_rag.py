import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()

print("==================================================")
print("   Week 7 — Day 5: Final RAG Deliverable")
print("==================================================\n")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom Chroma embedding function backed by the Gemini API."""

    def __init__(self, client):
        self.client = client

    def __call__(self, input: Documents) -> Embeddings:
        return [
            self.client.models.embed_content(model=EMBED_MODEL, contents=text)
            .embeddings[0].values
            for text in input
        ]


# ---------------------------------------------------------------------------
# 1. Document set
# ---------------------------------------------------------------------------
documents = [
    "Distributed systems integrate independent computers into a single coherent system, managing challenges like network partitions and latency.",
    "Algorithms like Lamport Timestamps, Vector Clocks, and Ricart-Agrawala maintain ordering and mutual exclusion across distributed nodes.",
    "Remote Procedure Call (RPC) protocols allow a program to execute code in another address space, hiding underlying network complexities.",
    "Consistency models dictate how state updates are propagated and observed across independent nodes in a system. Strong consistency guarantees every read sees the latest write, while eventual consistency trades that guarantee for higher availability.",
]

chroma_client = chromadb.PersistentClient(path="./chroma_db")
existing = [c.name for c in chroma_client.list_collections()]
if "week7_final" in existing:
    chroma_client.delete_collection(name="week7_final")

collection = chroma_client.get_or_create_collection(
    name="week7_final",
    embedding_function=GeminiEmbeddingFunction(client),
)
ids = [f"doc_{i}" for i in range(len(documents))]
collection.add(documents=documents, ids=ids)


# ---------------------------------------------------------------------------
# 2. Retrieval step
# ---------------------------------------------------------------------------
def retrieve(query: str, n_results: int = 2) -> list[str]:
    """RETRIEVE: fetch the most relevant chunks from the vector DB."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


# ---------------------------------------------------------------------------
# 3. Generation step (grounded prompt with citations)
# ---------------------------------------------------------------------------
def call_model(prompt: str, max_retries: int = 6) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt,
            )
            return response.text
        except errors.ClientError as e:
            raise RuntimeError(f"Non-retryable error calling '{CHAT_MODEL}': {e}") from e
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = min(2 ** attempt, 30)
            print(f"[Retry {attempt}/{max_retries}] {e}. Waiting {wait}s...")
            time.sleep(wait)


def generate_answer(query: str, chunks: list[str]) -> str:
    """GENERATE: answer strictly from retrieved context, with a citation."""
    numbered_context = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    prompt = f"""You are a careful assistant that answers ONLY using the numbered context below.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- If the context does not contain enough information to answer, respond exactly:
  "I don't have enough information to answer that."
- After your answer, on a new line, cite which chunk number(s) you used, like: Source: [1]

Context:
{numbered_context}

Question: {query}

Answer:"""
    return call_model(prompt)


# ---------------------------------------------------------------------------
# 4. Full RAG pipeline
# ---------------------------------------------------------------------------
def run_rag(query: str, n_results: int = 2):
    print(f"Question: {query}\n")
    chunks = retrieve(query, n_results=n_results)
    print("--- Retrieved context ---")
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] {c}")
    print()

    answer = generate_answer(query, chunks)
    print("--- Answer ---")
    print(answer)
    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 5. Example questions (deliverable: a few worked Q&As)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_rag("How do RPC protocols simplify writing distributed code?")
    run_rag("What is strong consistency?")
    run_rag("How exactly does a Lamport timestamp get incremented?")  # out-of-scope, should refuse
    run_rag("What's the capital of France?")  # unrelated, should refuse