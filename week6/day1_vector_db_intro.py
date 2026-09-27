import chromadb

print("==================================================")
print("   Week 6 — Day 1: Vector Database Intro (Chroma)")
print("==================================================\n")

# 1. Create a local, persistent Chroma client
#    (data is saved to disk in ./chroma_db so it survives between runs)
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Create (or get) a collection — think of this like a "table" for vectors
collection = client.get_or_create_collection(name="week6_intro")

# 3. Sample documents (same ones from Week 5 for later comparison)
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn canine leaps across a sleepy hound.",
    "Python is a popular programming language for data science.",
    "Artificial intelligence and machine learning are transforming tech.",
    "Delicious Nepalese momos are served with spicy tomato chutney.",
]
ids = [f"doc_{i}" for i in range(len(documents))]

# 4. Add documents — Chroma automatically embeds them using its
#    built-in default model (all-MiniLM-L6-v2), no API key needed
print("1. Adding documents to the collection...")
collection.add(documents=documents, ids=ids)
print(f"Added {len(documents)} documents.\n")

# 5. Query with a sample question
query = "Where can I find tasty dumplings in Nepal?"
print(f"2. Query: '{query}'\n")

results = collection.query(
    query_texts=[query],
    n_results=3,
)

# 6. Show ranked results
print("3. Top matches:")
for rank, (doc, distance) in enumerate(
    zip(results["documents"][0], results["distances"][0]), start=1
):
    print(f"[{rank}] Distance: {distance:.4f} | Document: '{doc}'")