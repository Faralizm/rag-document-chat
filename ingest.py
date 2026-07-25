import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#(from .env)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("XƏBƏRDARLIQ: OPENAI_API_KEY .env faylında tapılmadı!")

def ingest_and_chunk(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    PDF faylını oxuyur (Ingestion) və məntiqli parçalara bölür (Chunking).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Fayl tapılmadı: {pdf_path}")

    print(f"--> [{pdf_path}] faylı oxunur...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"--> Uğurla oxundu! Toplam səhifə sayı: {len(documents)}")


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"--> Sənəd məntiqli şəkildə {len(chunks)} chunk-a bölündü.")
    
    return chunks

if __name__ == "__main__":
    pdf_file = "sample.pdf" 

    try:
        chunks = ingest_and_chunk(pdf_file)
        
        # Nümunə üçün ilk 2 chunk-ı print edək
        print("\n--- NÜMUNƏ CHUNK (1) ---")
        print("MƏTN:", chunks[0].page_content)
        print("METADATA:", chunks[0].metadata)
        
    except Exception as e:
        print(f"Xəta baş verdi: {e}")