import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from ingest import load_pdf, chunk_documents
from embed import load_embedding_model, generate_embeddings
from vectorstore import build_faiss_index, similarity_search

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def build_prompt_with_sources(query, retrieved_chunks):

    context_text = "\n\n---\n\n".join(
        f"[source_id: {i + 1} | page: {r['chunk'].metadata.get('page', '?')}]\n{r['chunk'].page_content}"
        for i, r in enumerate(retrieved_chunks)
    )

    prompt = f"""You are a helpful assistant that answers questions using ONLY the context below.
If the answer is not contained in the context, say you don't know - do not make up an answer.

CONTEXT:
{context_text}

INSTRUCTION:
Answer the question using only the context above.
Respond ONLY with valid JSON in this exact format, no extra text:
{{
  "answer": "your answer here",
  "sources_used": [list of source_id numbers you actually used, e.g. 1, 3]
}}

QUESTION:
{query}"""
    return prompt


def ask_llm_with_sources(prompt):
    """Claude API-yə göndərir, JSON cavabı parse edir."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.content[0].text.strip()

    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        print("XƏBƏRDARLIQ: JSON parse alınmadı, xam mətn göstərilir:")
        print(raw_text)
        return None

    return parsed


def display_answer_with_sources(parsed, retrieved_chunks):
    print("--- Cavab ---")
    print(parsed["answer"])

    print("\n--- İstifadə olunan mənbələr ---")
    for source_id in parsed["sources_used"]:
        chunk_info = retrieved_chunks[source_id - 1]
        page = chunk_info["chunk"].metadata.get("page", "?")
        snippet = chunk_info["chunk"].page_content[:150]
        print(f"\nMənbə {source_id} (səhifə {page}):")
        print(f"{snippet}...")


if __name__ == "__main__":
    pdf_path = "sample.pdf"

    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)

    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    index = build_faiss_index(embeddings)

    query = "Bu sənəd nə haqqındadır?"  
    retrieved = similarity_search(query, model, index, chunks, top_k=3)

    prompt = build_prompt_with_sources(query, retrieved)
    parsed = ask_llm_with_sources(prompt)

    if parsed:
        display_answer_with_sources(parsed, retrieved)