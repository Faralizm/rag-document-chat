import faiss
import numpy as np
from ingest import load_pdf, chunk_documents
from embed import load_embedding_model, generate_embeddings


def build_faiss_index(embeddings):
    """
    Embedding vektorlarından FAISS index qurur.
    IndexFlatL2 -> Euclidean (L2) məsafəyə əsaslanan sadə, dəqiq axtarış.
    Kiçik/orta kolleksiyalar üçün sürətlidir.
    """
    dimension= embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    print(f"FAISS index quruldu: {index.ntotal} vektor, {dimension} ölçü")
    return index


def similarity_search(query, model, index, chunks, top_k=3):
    """
    Sorğunu (query) embedding-ə çevirir və indeksdə ən oxşar top_k chunk-ı tapır.
    """
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "distance": distances[0][rank],
        })
    return results


if __name__ == "__main__":
    pdf_path = "sample.pdf"

    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)

    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    index = build_faiss_index(embeddings)

    test_query = "Bu sənəd nə haqqındadır?"
    results = similarity_search(test_query, model, index, chunks, top_k=3)

    print(f"\n--- Sorğu: '{test_query}' üçün ən oxşar {len(results)} chunk ---")
    for i, r in enumerate(results):
        print(f"\n{i + 1}. Məsafə: {r['distance']:.4f}")
        print(f"Mətn: {r['chunk'].page_content[:200]}...")
        print(f"Mənbə: {r['chunk'].metadata}")