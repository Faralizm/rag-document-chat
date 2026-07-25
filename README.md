# RAG Document Chat — "Sənədlərinlə Danış"

A complete Retrieval-Augmented Generation (RAG) pipeline that loads a PDF document, splits it into chunks, generates embeddings, stores them in a FAISS vector index, retrieves relevant context for a query, and generates a source-attributed answer using the Claude API — with a guardrail against hallucinating answers not found in the document.

## Project structure

- `ingest.py` — Checkpoint 1: document ingestion + chunking
- `embed.py` — Checkpoint 2: embedding generation (reuses ingest.py)
- `vectorstore.py` — Checkpoint 3: FAISS vector storage + similarity search (reuses ingest.py, embed.py)
- `retrieval.py` — Checkpoint 4: retrieval + prompt construction, integrated with the Claude API
- `source_answer.py` — Checkpoint 5: source-attributed answer generation (structured JSON output with cited sources)
- `no_answer_test.py` — Checkpoint 6: "not in documents" handling, with a distance-based guardrail and a hallucination test case

## How to run it

1. Create and activate a virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
```

2. Install the dependencies:
```bash
   pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your own Anthropic API key:
```bash
   cp .env.example .env
```
   `.env` is listed in `.gitignore`, so it will never be pushed to GitHub.

4. Place a test PDF named `sample.pdf` in the project folder (you can use any PDF you like).

5. Run each stage in order:
```bash
   python ingest.py           # ingestion + chunking
   python embed.py            # embedding generation
   python vectorstore.py      # vector storage + similarity search
   python retrieval.py        # retrieval + prompt + LLM answer
   python source_answer.py    # answer with cited sources
   python no_answer_test.py   # hallucination guardrail test
```

## How it works

1. **Document ingestion**: `PyPDFLoader` reads the PDF page by page and loads it into Document format.
2. **Chunking strategy**: `RecursiveCharacterTextSplitter`
   - `chunk_size=1000` — enough context per chunk without exceeding token limits
   - `chunk_overlap=200` — key information at chunk boundaries isn't lost
   - `separators=["\n\n", "\n", " ", ""]` — splits by paragraph first, then sentence
3. **Embedding generation**: `sentence-transformers` with the local `all-MiniLM-L6-v2` model converts each chunk into a 384-dimensional vector. No API key or cost required.
4. **Vector storage + similarity search**: `FAISS` (`IndexFlatL2`) stores all chunk embeddings and performs exact L2-distance search to retrieve the most relevant chunks for a query.
5. **Retrieval + prompt construction**: retrieved chunks are assembled into a clearly labeled CONTEXT section, kept separate from the INSTRUCTION and QUESTION, so the model can distinguish retrieved information from its task.
6. **Source-attributed answers**: the model returns structured JSON (`answer` + `sources_used`) so every claim can be traced back to a specific chunk and page.
7. **Hallucination guardrail**: if the closest retrieved chunk exceeds a distance threshold, the system returns "not answerable from documents" without calling the LLM. The prompt also explicitly instructs the model to admit when it doesn't know rather than fabricate an answer.

## Security

API keys are loaded from a `.env` file using `python-dotenv`. `.env` is included in `.gitignore`, so no sensitive credentials will ever be pushed to GitHub. A `.env.example` file is included in the repo as a setup template, with no real key inside.