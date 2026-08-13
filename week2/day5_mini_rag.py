import math
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


# --- 1. UTILITY FUNCTIONS ---
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculates cosine similarity between two numerical vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def get_embedding(client: genai.Client, text: str) -> list[float]:
    """Retrieves text embedding vector with model fallback support."""
    candidate_models = [
        "text-embedding-004",
        "gemini-embedding-001",
        "models/text-embedding-004",
        "models/gemini-embedding-001",
    ]

    for model_name in candidate_models:
        try:
            res = client.models.embed_content(
                model=model_name,
                contents=text,
            )
            if hasattr(res, "embedding") and res.embedding:
                return res.embedding.values
            elif hasattr(res, "embeddings") and res.embeddings:
                return res.embeddings[0].values
        except Exception:
            continue

    raise RuntimeError("Failed to fetch embedding with available models.")


def chunk_text(text: str, chunk_size: int = 35, overlap: int = 10) -> list[str]:
    """Splits a long string into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# --- 2. RAG PIPELINE CLASS ---
class MiniRAG:

    def __init__(self, client: genai.Client):
        self.client = client
        self.chunks = []
        self.chunk_embeddings = []

    def index_document(self, document_text: str):
        """Clean, chunk, and embed a source document."""
        print("📄 Indexing document into vector memory...")
        self.chunks = chunk_text(document_text)

        for idx, chunk in enumerate(self.chunks, start=1):
            emb = get_embedding(self.client, chunk)
            self.chunk_embeddings.append(emb)
            time.sleep(1)  # Rate limit pause

        print(f"✅ Indexed {len(self.chunks)} chunks successfully!\n")

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        """Find the top-K most relevant chunks for a given query."""
        query_emb = get_embedding(self.client, query)

        scores = []
        for chunk, emb in zip(self.chunks, self.chunk_embeddings):
            sim = cosine_similarity(query_emb, emb)
            scores.append((chunk, sim))

        # Sort by similarity score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in scores[:top_k]]

    def answer_question(self, query: str) -> str:
        """Complete RAG pipeline: Retrieve relevant context and generate grounded answer."""
        print(f"❓ User Query: \"{query}\"")

        # Step 1: Retrieve context
        retrieved_chunks = self.retrieve(query, top_k=2)
        context = "\n---\n".join(retrieved_chunks)

        print("\n🔍 Retrieved Context:")
        print(context)
        print("-" * 50)

        # Step 2 & 3: Augment prompt and Generate answer
        config = types.GenerateContentConfig(
            system_instruction=(
                "You are an accurate grounded assistant. Answer the user's question "
                "ONLY using the provided context. If the answer is not in the context, say "
                "'I cannot answer based on the provided information.'"
            ),
            temperature=0.1,
        )

        prompt = f"Context:\n{context}\n\nQuestion: {query}"

        response = self.client.models.generate_content(
            model="gemini-flash-latest", contents=prompt, config=config
        )

        return response.text.strip()


# --- 3. MAIN EXECUTION ---
def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)

    # Knowledge Base Document
    knowledge_base = """
    Distributed systems combine independent computers to function as a single system.
    Key problems include network latency, clock drift, and partitions.
    
    To maintain mutual exclusion across nodes without a central authority, the Ricart-Agrawala 
    algorithm is used. It relies on timestamped request messages to resolve conflicting access requests.
    
    For synchronization, Cristians algorithm relies on a central time server, while Lamport Timestamps
    provide logical ordering of events using scalar counters across distributed nodes.
    
    Remote Procedure Call (RPC) abstracts network communication, allowing procedures to be executed 
    in remote address spaces as if they were local function calls.
    """

    rag_system = MiniRAG(client)

    # 1. Index document
    rag_system.index_document(knowledge_base)

    # 2. Query knowledge base
    query = "How does the Ricart-Agrawala algorithm work?"
    answer = rag_system.answer_question(query)

    print("\n🤖 Final Grounded Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()