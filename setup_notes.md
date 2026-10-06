# Week 1 - Setup Notes

### Day 1
- Set up venv, cloned repo, wrote basic list/dict examples in `basics.py`.

### Day 2
- Fixed Git author config. Practiced loops, conditionals, and functions (`loops_conditionals.py`, `functions.py`).

### Day 3
- Practiced string methods, file I/O, JSON parsing, and `try-except` error handling.

### Day 4
- Used `requests` to fetch a REST API and saved output to `api_response.json`.


# Week 2 - Setup Notes

### Day 1
- Set up `.env` + `.gitignore`, installed `google-genai`, ran first Gemini call (`gemini-2.5-flash`).

### Day 2
- Explored `temperature`, `max_output_tokens`, and `system_instruction` via `types.GenerateContentConfig`.

### Day 3
- Built word-level sliding-window chunking (`chunk_size=35`, `overlap=10`).

### Day 4
- Generated embeddings and implemented cosine similarity from scratch for semantic ranking.

### Day 5
- Built `MiniRAG`: chunking + embeddings + retrieval + generation in one pipeline.

### Day 6
- Refactored Week 2 code, standardized error handling, updated README.


# Week 3 - Setup Notes

### Day 1
- Built streaming responses with `generate_content_stream()`.

### Day 2
- Added system prompts for persona/behavior control.

### Day 3
- Built multi-turn chat with `client.chats.create()` and a `history` command.

### Day 4
- Tuned `temperature`, `top_p`, `top_k`, `max_output_tokens`.

### Day 5
- Added error handling for API failures, quota limits, and interrupts.

### Day 6
- Combined everything into one final CLI chatbot script.


# Week 4 - Setup Notes

### Day 1-3
- Practiced zero-shot, few-shot, and chain-of-thought prompting; validated structured JSON output.

### Day 4
- Mini project: a text tool for summarizing/extracting info using the Gemini API.


# Week 5 - Setup Notes

### Day 1
- Built semantic search using `gemini-embedding-001` and manual cosine similarity.
- Fixed model-name issues (Vertex AI names don't work on the Gemini API).


# Week 6 - Setup Notes

### Day 1
- Set up ChromaDB locally, ran first add/query test.

### Day 2
- Implemented fixed-size chunking with overlap, stored chunks with deterministic IDs.

### Day 3
- Swapped Chroma's default embedder for a custom Gemini-based one.

### Day 4
- Tested multiple queries and compared chunk sizes (300 vs 150 chars).

### Day 5
- Compared manual cosine-similarity approach vs ChromaDB — identical ranking, DB adds persistence.


# Week 7 - Setup Notes

### Day 1
- Built the first working RAG loop: retrieve chunks, then generate an answer from them.

### Day 2
- Added citations and a strict refusal rule for insufficient context.

### Day 3
- Tested fully-answerable, partially-relevant, and unrelated questions — model refused correctly on all edge cases.

### Day 4
- Compared a strict prompt vs a weak one — weak prompt reliably caused hallucination (Lamport clocks, CAP theorem answered from training data, not context).

### Day 5
- Finalized the RAG script with a clean set of example Q&As as the deliverable.


# Week 8 - Setup Notes

### Day 1
- Wrapped the RAG pipeline in a basic Streamlit UI.

### Day 2
- Added file upload + chunking so users can load their own documents.

### Day 3
- Added a relevance threshold and error handling for empty files/questions and API failures.

### Day 4
- Rebuilt as a proper chat UI with history, plus custom branding and color theme.

### Day 5
- Ran final tests across question types and wrote the app's README.

# Week 9 - Setup Notes

### Day 1
- Defined function calling tools with type annotations, parameter schemas, and descriptive docstrings.

### Day 2
- Implemented automatic function invocation and execution loops using Gemini tool bindings.

### Day 3
- Added edge-case handling for tool failures, empty database lookups, and numeric domain errors.

### Day 4
- Built multi-tool agent chaining where outputs from initial tool calls pass directly into subsequent tools.

### Day 5
- Developed a head-to-head synthesis benchmark comparing plain parametric prompting versus deterministic tool execution.

### Day 6
- Standardized model fallback logic across all scripts, refactored code for production readiness, and completed the repository README.