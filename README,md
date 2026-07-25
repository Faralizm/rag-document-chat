# Checkpoint 1 — Document Ingestion + Chunking

A simple pipeline that loads a PDF document and splits it into logically sized chunks.

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

3. Copy `.env.example` to `.env` and add your own API key:
```bash
   cp .env.example .env
```
   `.env` is listed in `.gitignore`, so it will never be pushed to GitHub.

4. Place a test PDF named `sample.pdf` in the project folder (you can use any PDF you like).

5. Run the script:
```bash
   python ingest.py
```

## How it works

1. **Document ingestion**: `PyPDFLoader` reads the PDF page by page and loads it into Document format.
2. **Chunking strategy**: `RecursiveCharacterTextSplitter`
   - `chunk_size=1000` — gives enough context per chunk without exceeding token limits
   - `chunk_overlap=200` — ensures key information at chunk boundaries isn't lost
   - `separators=["\n\n", "\n", " ", ""]` — splits by paragraph first, then sentence, to avoid cutting words in half

## Security

API keys are loaded from a `.env` file using `python-dotenv`. `.env` is included in `.gitignore`, so no sensitive credentials will ever be pushed to GitHub. A `.env.example` file is included in the repo as a setup template, with no real key inside.