from ingest import load_pdf, chunk_documents
from embed import load_embedding_model, generate_embeddings
from vectorstore import build_faiss_index, similarity_search
from source_answer import build_prompt_with_sources, ask_llm_with_sources, display_answer_with_sources

DISTANCE_THRESHOLD = 1.5


def answer_with_guardrail(query, model, index, chunks, top_k=3):
    retrieved = similarity_search(query, model, index, chunks, top_k=top_k)

    best_distance = retrieved[0]["distance"]
    if best_distance > DISTANCE_THRESHOLD:
        print(f"(Ən yaxın chunk məsafəsi {best_distance:.2f} > threshold {DISTANCE_THRESHOLD}, LLM çağırılmır)")
        return {"answer": "Bu sual sənədlərdə cavablandırılmır.", "sources_used": []}

    prompt = build_prompt_with_sources(query, retrieved)
    parsed = ask_llm_with_sources(prompt)
    return parsed, retrieved


if __name__ == "__main__":
    pdf_path = "sample.pdf"

    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)

    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    index = build_faiss_index(embeddings)

    # TEST 1
    real_query = "Bu sənəd nə haqqındadır?"  # öz sənədinə uyğun sualla dəyiş
    print(f"=== TEST 1 (real sual): {real_query} ===")
    result = answer_with_guardrail(real_query, model, index, chunks)
    if isinstance(result, tuple):
        parsed, retrieved = result
        if parsed:
            display_answer_with_sources(parsed, retrieved)
    else:
        print(result["answer"])

    # TEST 2
    fake_query = "Marsda həyat forması varmı və bu sənəddə bundan bəhs olunurmu?"
    print(f"\n\n=== TEST 2 (sənəddə olmayan sual): {fake_query} ===")
    result = answer_with_guardrail(fake_query, model, index, chunks)
    if isinstance(result, tuple):
        parsed, retrieved = result
        if parsed:
            display_answer_with_sources(parsed, retrieved)
    else:
        print(result["answer"])