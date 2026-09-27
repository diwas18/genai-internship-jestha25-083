import os
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()

print("==================================================")
print("   Week 6 — Day 4: Query Practice & Tuning")
print("==================================================\n")

EMBED_MODEL = "gemini-embedding-001"


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in .env")
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings


document = """
Distributed systems integrate independent computers into a single coherent
system, managing challenges like network partitions and latency. Because
nodes can fail independently and messages can be delayed or lost, designing
correct distributed systems is notoriously hard.

Algorithms like Lamport Timestamps, Vector Clocks, and Ricart-Agrawala
maintain ordering and mutual exclusion across distributed nodes. These
algorithms allow processes running on different machines to agree on the
relative order of events without a shared global clock.

Remote Procedure Call (RPC) protocols allow a program to execute code in
another address space, hiding underlying network complexities. This lets
developers write distributed code that looks similar to a normal local
function call, even though it may cross machine boundaries.

Consistency models dictate how state updates are propagated and observed
across independent nodes in a system. Strong consistency guarantees every
read sees the latest write, while eventual consistency trades that
guarantee for higher availability and lower latency.
"""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_collection(client, name, chunk_size, overlap):
    """(Re)builds a Chroma collection with a given chunking config."""
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name=name)

    collection = client.get_or_create_collection(
        name=name,
        embedding_function=GeminiEmbeddingFunction(),
    )
    chunks = chunk_text(document, chunk_size=chunk_size, overlap=overlap)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return collection, chunks


def run_queries(collection, queries, n_results=2):
    for query in queries:
        print(f"Query: '{query}'")
        results = collection.query(query_texts=[query], n_results=n_results)
        for rank, (doc, distance) in enumerate(
            zip(results["documents"][0], results["distances"][0]), start=1
        ):
            preview = doc[:100].replace("\n", " ")
            print(f"  [{rank}] Distance: {distance:.4f} | '{preview}...'")
        print()


client = chromadb.PersistentClient(path="./chroma_db")

# --- Round 1: same chunking as Day 3 (chunk_size=300, overlap=50) ---
print("### Round 1: chunk_size=300, overlap=50 ###\n")
collection_a, chunks_a = build_collection(client, "week6_tune_a", 300, 50)
print(f"({len(chunks_a)} chunks)\n")

queries = [
    "How do processes agree on event ordering without a shared clock?",
    "What is RPC used for?",
    "What's the difference between strong and eventual consistency?",
    "Why are distributed systems hard to build correctly?",
]

run_queries(collection_a, queries)

# --- Round 2: smaller, tighter chunks ---
print("### Round 2: chunk_size=150, overlap=30 ###\n")
collection_b, chunks_b = build_collection(client, "week6_tune_b", 150, 30)
print(f"({len(chunks_b)} chunks)\n")

run_queries(collection_b, queries)