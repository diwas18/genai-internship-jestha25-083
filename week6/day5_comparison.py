import os
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()

print("==================================================")
print("   Week 6 — Day 5: Manual vs. Vector DB Comparison")
print("==================================================\n")

EMBED_MODEL = "gemini-embedding-001"
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai_client = genai.Client(api_key=api_key)

# Same document set as Week 5, so the comparison is fair
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn canine leaps across a sleepy hound.",
    "Python is a popular programming language for data science.",
    "Artificial intelligence and machine learning are transforming tech.",
    "Delicious Nepalese momos are served with spicy tomato chutney.",
]
query = "Where can I find tasty dumplings in Nepal?"


# ---------- APPROACH 1: Week 5 manual cosine similarity ----------

def cosine_similarity(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding(text):
    response = genai_client.models.embed_content(model=EMBED_MODEL, contents=text)
    return response.embeddings[0].values


print("### Approach 1: Manual (Week 5 style) ###\n")
start = time.time()

doc_embeddings = [get_embedding(doc) for doc in documents]
query_embedding = get_embedding(query)

manual_results = []
for doc, emb in zip(documents, doc_embeddings):
    score = cosine_similarity(query_embedding, emb)
    manual_results.append((doc, score))
manual_results.sort(key=lambda x: x[1], reverse=True)

manual_time = time.time() - start

for rank, (doc, score) in enumerate(manual_results, start=1):
    print(f"[{rank}] Similarity: {score:.4f} | '{doc}'")
print(f"\nTime taken: {manual_time:.2f}s "
      f"({len(documents) + 1} embedding calls, all held in memory, no storage)\n")


# ---------- APPROACH 2: ChromaDB vector database ----------

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client):
        self.client = client

    def __call__(self, input: Documents) -> Embeddings:
        return [
            self.client.models.embed_content(model=EMBED_MODEL, contents=text)
            .embeddings[0].values
            for text in input
        ]


print("### Approach 2: ChromaDB (Vector Database) ###\n")
start = time.time()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
existing = [c.name for c in chroma_client.list_collections()]
if "week6_final" in existing:
    chroma_client.delete_collection(name="week6_final")

collection = chroma_client.get_or_create_collection(
    name="week6_final",
    embedding_function=GeminiEmbeddingFunction(genai_client),
)
ids = [f"doc_{i}" for i in range(len(documents))]
collection.add(documents=documents, ids=ids)

results = collection.query(query_texts=[query], n_results=len(documents))
db_time = time.time() - start

for rank, (doc, distance) in enumerate(
    zip(results["documents"][0], results["distances"][0]), start=1
):
    print(f"[{rank}] Distance: {distance:.4f} | '{doc}'")
print(f"\nTime taken: {db_time:.2f}s "
      f"(embeddings persisted to disk in ./chroma_db, reusable across runs)\n")


# ---------- Side-by-side summary ----------

print("### Summary ###\n")
print(f"{'Rank':<6}{'Manual (Week 5)':<45}{'ChromaDB':<45}")
for i in range(len(documents)):
    m_doc = manual_results[i][0][:35]
    d_doc = results["documents"][0][i][:35]
    print(f"{i+1:<6}{m_doc:<45}{d_doc:<45}")

print(f"\nManual approach time:  {manual_time:.2f}s")
print(f"ChromaDB approach time: {db_time:.2f}s")