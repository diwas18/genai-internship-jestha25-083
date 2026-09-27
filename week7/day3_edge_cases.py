import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()

print("==================================================")
print("   Week 7 — Day 3: Testing Grounding Edge Cases")
print("==================================================\n")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"  # pinned model, swapped off the overloaded 'latest' alias

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
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


def generate_answer(query: str, context_chunks: list[str], max_retries: int = 6) -> str:
    numbered_context = "\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )

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


def run_rag(label: str, query: str, n_results: int = 2):
    print(f"[{label}] Question: {query}\n")
    chunks = retrieve(query, n_results=n_results)
    print("--- Retrieved context ---")
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}] {chunk}")
    print()

    answer = generate_answer(query, chunks)
    print("--- Generated answer ---")
    print(answer)
    print("\n" + "=" * 60 + "\n")


# 1. Fully answerable (control)
run_rag(
    "FULLY ANSWERABLE",
    "How do RPC protocols simplify writing distributed code?"
)

# 2. Partially relevant — docs mention Lamport/Vector Clocks generally,
#    but don't explain HOW Lamport timestamps specifically work
run_rag(
    "PARTIALLY RELEVANT",
    "How exactly does a Lamport timestamp get incremented?"
)

# 3. Adjacent but wrong — docs discuss consistency models, but never
#    mention CAP theorem specifically
run_rag(
    "ADJACENT BUT WRONG",
    "How does the CAP theorem relate to consistency models?"
)

# 4. Totally unrelated (control)
run_rag(
    "TOTALLY UNRELATED",
    "What's the capital of France?"
)