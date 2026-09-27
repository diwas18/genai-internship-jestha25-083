import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-flash-lite-latest"
RELEVANCE_THRESHOLD = 1.5


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client):
        self.client = client

    def __call__(self, input: Documents) -> Embeddings:
        return [
            self.client.models.embed_content(model=EMBED_MODEL, contents=text)
            .embeddings[0].values
            for text in input
        ]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


@st.cache_resource
def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Check your .env file.")
        st.stop()
    return genai.Client(api_key=api_key)


def build_collection(client, chunks: list[str]):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    existing = [c.name for c in chroma_client.list_collections()]
    if "week8_app" in existing:
        chroma_client.delete_collection(name="week8_app")

    collection = chroma_client.get_or_create_collection(
        name="week8_app",
        embedding_function=GeminiEmbeddingFunction(client),
    )
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return collection


def retrieve(collection, query: str, n_results: int = 2):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0], results["distances"][0]


def generate_answer(client, query: str, chunks: list[str], max_retries: int = 6):
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

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
            return response.text, None
        except errors.ClientError:
            return None, "The model rejected this request. Try a different question or check your API setup."
        except Exception:
            if attempt == max_retries:
                return None, "The AI service is temporarily unavailable after several retries. Please try again in a minute."
            time.sleep(min(2 ** attempt, 30))


def answer_question(client, collection, query: str) -> dict:
    try:
        chunks, distances = retrieve(collection, query)
    except Exception as e:
        return {"query": query, "answer": f"Couldn't search the document: {e}",
                "chunks": [], "distances": []}

    if chunks and min(distances) > RELEVANCE_THRESHOLD:
        return {
            "query": query,
            "answer": "This document doesn't seem to contain information relevant to that question.",
            "chunks": chunks,
            "distances": distances,
        }

    if not chunks:
        return {"query": query, "answer": "No context could be retrieved.",
                "chunks": [], "distances": []}

    answer, error = generate_answer(client, query, chunks)
    return {
        "query": query,
        "answer": error if error else answer,
        "chunks": chunks,
        "distances": distances,
    }


# ---------------- Page config + theme ----------------

st.set_page_config(page_title="Ask Your Documents", page_icon="📖", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

:root {
    --paper: #FAF9F6;
    --ink: #12343B;
    --teal: #1F6F78;
    --amber: #C98A3E;
    --slate: #5C6670;
}

.stApp {
    background-color: var(--paper);
    font-family: 'Inter', sans-serif;
}

/* Header block */
.app-header {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    margin-bottom: 0.5rem;
}
.app-header .role-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--teal);
    letter-spacing: 0.02em;
}
.app-header .app-title {
    font-family: 'Fraunces', serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.15;
}
.app-header .by-line {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: var(--slate);
}
.app-subcaption {
    font-family: 'Inter', sans-serif;
    color: var(--slate);
    font-size: 0.95rem;
    margin-bottom: 1.4rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F1EFE9;
    border-right: 1px solid #E3E0D6;
}
section[data-testid="stSidebar"] h2 {
    font-family: 'Fraunces', serif;
    color: var(--ink);
}

/* Buttons */
.stButton > button {
    background-color: var(--teal);
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 500;
    padding: 0.5rem 1.1rem;
}
.stButton > button:hover {
    background-color: var(--ink);
    color: white;
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
    background-color: white;
    border-radius: 12px;
    padding: 0.4rem 0.2rem;
    border: 1px solid #ECE9E1;
}

/* Citation styling inside expanders */
.stExpander {
    border: 1px solid #ECE9E1 !important;
    border-radius: 8px !important;
}

/* Footer */
.app-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #E3E0D6;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: var(--slate);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div class="role-label">GENERATIVE AI INTERNSHIP · WEEK 8</div>
    <div class="app-title">Ask Your Documents</div>
    <div class="by-line">Built by Diwas Sigdel</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="app-subcaption">Upload a document, then ask questions grounded strictly in its content.</div>',
    unsafe_allow_html=True,
)

client = get_client()

if "collection" not in st.session_state:
    st.session_state.collection = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("## Document")
    uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])

    if uploaded_file is not None and st.button("Process document"):
        try:
            raw_bytes = uploaded_file.read()
            if not raw_bytes:
                st.error("That file is empty.")
            else:
                text = raw_bytes.decode("utf-8", errors="ignore").strip()
                if len(text) < 20:
                    st.error("Not enough readable text in that file.")
                else:
                    with st.spinner("Chunking and embedding..."):
                        chunks = chunk_text(text)
                        collection = build_collection(client, chunks)
                    st.session_state.collection = collection
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.history = []
                    st.success(f"Loaded {len(chunks)} chunks.")
        except UnicodeDecodeError:
            st.error("Couldn't read that file as text.")
        except Exception as e:
            st.error(f"Error processing document: {e}")

    if st.session_state.collection is not None:
        st.markdown(f"**Active:** {st.session_state.chunk_count} chunks")
        if st.button("Clear conversation"):
            st.session_state.history = []
            st.rerun()

# Main chat area
if st.session_state.collection is None:
    st.info("Upload a document in the sidebar to get started.")
else:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["chunks"]:
                with st.expander("Retrieved context"):
                    for i, (c, d) in enumerate(zip(turn["chunks"], turn["distances"]), start=1):
                        st.markdown(f"**[{i}]** (distance: {d:.3f}) {c}")

    query = st.chat_input("Ask a question about the document...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question(client, st.session_state.collection, query)
            st.write(result["answer"])
            if result["chunks"]:
                with st.expander("Retrieved context"):
                    for i, (c, d) in enumerate(zip(result["chunks"], result["distances"]), start=1):
                        st.markdown(f"**[{i}]** (distance: {d:.3f}) {c}")

        st.session_state.history.append(result)

st.markdown(
    '<div class="app-footer">Generative AI Internship Program · Week 8 Deliverable · Diwas Sigdel</div>',
    unsafe_allow_html=True,
)