# RAG Document Chat — Checkpoints 1–3

A pipeline that loads a PDF document, splits it into chunks, generates embeddings, and stores them in a FAISS vector index for similarity search.

## Project structure

- `ingest.py` — Checkpoint 1: document ingestion + chunking
- `embed.py` — Checkpoint 2: embedding generation (reuses ingest.py)
- `vectorstore.py` — Checkpoint 3: FAISS vector storage + similarity search (reuses ingest.py and embed.py)

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

3. Copy `.env.example` to `.env` and add your own API key (not required for the current checkpoints, but set up for future use):
```bash
   cp .env.example .env
```
   `.env` is listed in `.gitignore`, so it will never be pushed to GitHub.

4. Place a test PDF named `sample.pdf` in the project folder (you can use any PDF you like).

5. Run each stage:
```bash
   python ingest.py         # ingestion + chunking
   python embed.py          # embedding generation
   python vectorstore.py    # vector storage + similarity search
```

## How it works

1. **Document ingestion**: `PyPDFLoader` reads the PDF page by page and loads it into Document format.
2. **Chunking strategy**: `RecursiveCharacterTextSplitter`
   - `chunk_size=1000` — gives enough context per chunk without exceeding token limits
   - `chunk_overlap=200` — ensures key information at chunk boundaries isn't lost
   - `separators=["\n\n", "\n", " ", ""]` — splits by paragraph first, then sentence, to avoid cutting words in half
3. **Embedding generation**: `sentence-transformers` with the local `all-MiniLM-L6-v2` model converts each chunk into a 384-dimensional vector. No API key or cost required.
4. **Vector storage + similarity search**: `FAISS` (`IndexFlatL2`) stores all chunk embeddings and performs exact L2-distance search to retrieve the most relevant chunks for a given query.

## Security

API keys are loaded from a `.env` file using `python-dotenv`. `.env` is included in `.gitignore`, so no sensitive credentials will ever be pushed to GitHub. A `.env.example` file is included in the repo as a setup template, with no real key inside.