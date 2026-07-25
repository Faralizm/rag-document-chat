from sentence_transformers import SentenceTransformer

from ingest import load_pdf, chunk_documents

#   from langchain_openai import OpenAIEmbeddings
#   embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
#   vectors = embedding_model.embed_documents([c.page_content for c in chunks])
MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """Local embedding modelini yükləyir (ilk dəfə internetdən endirilir, sonra cache-lənir)."""
    model = SentenceTransformer(MODEL_NAME)
    return model


def generate_embeddings(chunks, model):
    """
    Hər chunk-ın mətnini rəqəm vektoruna (embedding) çevirir.
    Bu vektorlar sonrakı checkpoint-də (vektor saxlama + oxşarlıq axtarışı) istifadə olunacaq.
    """
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings, hər biri {embeddings.shape[1]} ölçülü vektor")
    return embeddings


if __name__ == "__main__":
    pdf_path = "sample.pdf"

    # Checkpoint 1-dəki funksiyaları burda təkrar istifadə edirik
    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)

    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    # Yoxlama üçün ilk embedding-in ilk 5 dəyərini çap edirik
    print("\n--- İlk embedding nümunəsi (ilk 5 dəyər) ---")
    print(embeddings[0][:5])
    print(f"\nTam ölçü: {embeddings[0].shape}")