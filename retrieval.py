import os

from anthropic import Anthropic
from dotenv import load_dotenv
from ingest import load_pdf, chunk_documents
from embed import load_embedding_model, generate_embeddings
from vectorstore import build_faiss_index, similarity_search

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def build_prompt(query, retrieved_chunks):
    
    context_text = "\n\n---\n\n".join(
        f"[Mənbə {i + 1} | səhifə {r['chunk'].metadata.get('page', '?')}]\n{r['chunk'].page_content}"
        for i, r in enumerate(retrieved_chunks)
    )

    prompt = f"""You are a helpful assistant that answers questions using ONLY the context below.
If the answer is not contained in the context, say you don't know - do not make up an answer.

CONTEXT:
{context_text}

INSTRUCTION:
Answer the following question using only the context above. Cite which source number(s) you used.

QUESTION:
{query}"""
    return prompt


def ask_llm(prompt):
    """Claude API-yə prompt-u göndərir və cavabı qaytarır."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    pdf_path = "sample.pdf"

    # Əvvəlki checkpoint-lərdəki funksiyaları təkrar istifadə edirik
    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)

    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    index = build_faiss_index(embeddings)

    query = "Bu sənəd nə haqqındadır?"  # öz sualınla dəyiş
    retrieved = similarity_search(query, model, index, chunks, top_k=3)

    prompt = build_prompt(query, retrieved)
    print("--- Qurulmuş prompt ---")
    print(prompt)

    answer = ask_llm(prompt)
    print("\n--- LLM cavabı ---")
    print(answer)