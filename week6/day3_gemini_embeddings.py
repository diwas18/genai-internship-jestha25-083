import os
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()

print("==================================================")
print("   Week 6 — Day 3: Gemini Embeddings in ChromaDB")
print("==================================================\n")

EMBED_MODEL = "gemini-embedding-001"


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom Chroma embedding function that calls the Gemini API."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in .env")
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        # input is a list of strings (chunks or queries) — embed each one
        embeddings = []
        for text in input:
            response = self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings


# 1. Reuse the same document + chunker from Day 2
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


chunks = chunk_text(document, chunk_size=300, overlap=50)
print(f"1. Split document into {len(chunks)} chunks.\n")

# 2. Create a Chroma collection that uses Gemini embeddings
client = chromadb.PersistentClient(path="./chroma_db")

existing = [c.name for c in client.list_collections()]
if "week6_gemini" in existing:
    client.delete_collection(name="week6_gemini")

collection = client.get_or_create_collection(
    name="week6_gemini",
    embedding_function=GeminiEmbeddingFunction(),
)

# 3. Store chunks (Chroma will call GeminiEmbeddingFunction automatically)
ids = [f"chunk_{i}" for i in range(len(chunks))]
print("2. Embedding and storing chunks via Gemini API (may take a few seconds)...")
collection.add(documents=chunks, ids=ids)
print(f"Stored {len(chunks)} chunks using '{EMBED_MODEL}'.\n")

# 4. Query — Chroma will embed the query with the same Gemini function
query = "How do processes agree on event ordering without a shared clock?"
print(f"3. Test query: '{query}'\n")

results = collection.query(query_texts=[query], n_results=2)

print("4. Top matching chunks (Gemini embeddings):")
for rank, (doc, distance) in enumerate(
    zip(results["documents"][0], results["distances"][0]), start=1
):
    print(f"[{rank}] Distance: {distance:.4f}\n    '{doc}'\n")