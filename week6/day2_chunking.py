import chromadb

print("==================================================")
print("   Week 6 — Day 2: Chunking & Storage")
print("==================================================\n")

# 1. A longer sample document (swap this for a real one — an article,
#    a wiki page, your own notes, etc.)
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
    """Splits text into overlapping fixed-size chunks.

    chunk_size: max characters per chunk
    overlap: characters repeated between consecutive chunks, to avoid
             losing context at chunk boundaries
    """
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step forward, but overlap a bit
    return chunks


# 2. Chunk the document
chunks = chunk_text(document, chunk_size=300, overlap=50)
print(f"1. Split document into {len(chunks)} chunks:\n")
for i, c in enumerate(chunks):
    print(f"  Chunk {i}: '{c[:80]}...'")
print()

# 3. Set up Chroma — clear the collection first so reruns don't
#    conflict on duplicate IDs
client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection(name="week6_chunks") if "week6_chunks" in [
    c.name for c in client.list_collections()
] else None
collection = client.get_or_create_collection(name="week6_chunks")

# 4. Store chunks with deterministic IDs
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(documents=chunks, ids=ids)
print(f"2. Stored {len(chunks)} chunks in the 'week6_chunks' collection.\n")

# 5. Quick test query to confirm chunk-level retrieval works
query = "How do processes agree on event ordering without a shared clock?"
print(f"3. Test query: '{query}'\n")

results = collection.query(query_texts=[query], n_results=2)

print("4. Top matching chunks:")
for rank, (doc, distance) in enumerate(
    zip(results["documents"][0], results["distances"][0]), start=1
):
    print(f"[{rank}] Distance: {distance:.4f}\n    '{doc}'\n")