# Week 1 - Setup Notes
# Day 1

- Python and VS Code were already installed
- Created and activated a virtual environment (venv)
- Cloned the repo locally and set up the week1 folder
- Wrote basic Python examples using lists and dictionaries in basics.py
- Ran basics.py successfully with no errors


# Day 2 

- Aim: Learning Python functions, control loops, and dictionary collections
- Resolved Git author configuration to properly attribute commits to `diwas18`
- Created `loops_conditionals.py` to practice `if-elif-else` branches, `for` loops, and `while` loops
- Created `functions.py` featuring modular functions (`greet_user`, `calculate_grade`, `summarize_scores`) using dictionary returns and list operations
- Verified output in the local virtual environment and pushed changes to GitHub

# Day 3

- Aim: Practicing core Python basics: string methods, file I/O, error handling, and JSON
- Practiced string manipulation (`strip()`, `lower()`) and text file I/O (`open()`, `read()`, `write()`)
- Applied JSON parsing (`json.dump()`, `json.load()`) to serialize and deserialize structured data
- Implemented robust error handling using `try-except` blocks for missing files (`FileNotFoundError`) and invalid JSON (`JSONDecodeError`)
- Verified all scripts locally and pushed changes to GitHub


### Day 4: API Requests & Data Processing
- Installed `requests` package to fetch external REST API data.
- Parsed user records and saved output to `week1/api_response.json`.



# Week 2 - Setup Notes

### Day 1: Gemini API & Security
- Installed `python-dotenv` and `google-genai`.
- Secured API key in `.env` and updated `.gitignore` to prevent leaks.
- Created `week2/day1_llm_intro.py` using `genai.Client` and `gemini-2.5-flash`.
- Managed rate limits (`429`) with `time.sleep()` delay.
- Successfully generated first LLM completion in terminal and pushed to GitHub.

### Day 2: Model Configuration & Prompting
- Explored parameter tuning including `temperature`, `max_output_tokens`, and `system_instruction`.
- Built `week2/day2_config.py` using `types.GenerateContentConfig` to control output randomness and constraints.
- Learned how system prompts dictate persona and behavior for LLM responses.

### Day 3: Text Preprocessing & Chunking
- Implemented text cleaning routines for raw document data.
- Created word-level sliding-window chunking (`chunk_size=35`, `overlap=10`) in `week2/day3_chunking.py`.
- Preserved semantic context across chunk boundaries for downstream RAG use.

### Day 4: Vector Embeddings & Semantic Search
- Converted document text chunks into high-dimensional vector representations using Gemini embedding models.
- Implemented a custom mathematical `cosine_similarity()` function in `week2/day4_vector_embeddings.py`.
- Added model fallback handling and verified semantic ranking against user queries.

### Day 5: Mini-RAG System Implementation
- Integrated chunking, vector embeddings, cosine search, and Gemini generation into `week2/day5_mini_rag.py`.
- Built an end-to-end `MiniRAG` pipeline featuring document indexing, top-K retrieval, and prompt augmentation.
- Configured strict system instructions (`temperature=0.1`) to generate grounded, non-hallucinated answers from context.

### Day 6: Code Refactoring & Repository Documentation
- Refactored `week2/` codebase to standardize error handling and model fallback sequences across modules.
- Updated main `README.md` with detailed Week 2 architecture breakdowns and module summaries.
- Verified all committed scripts pass standalone execution tests and pushed clean commits to GitHub.