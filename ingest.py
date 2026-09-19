"""
Loads the Agentic AI eBook PDF, splits it into overlapping chunks,
generates embeddings, and persists a FAISS vector index + chunk metadata
to vector_store.

Run this once (whenever source PDF changes):
"""
import os
import pickle
import config

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy



def load_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Could not find PDF at '{pdf_path}'. "
            f"Place 'Ebook-Agentic-AI.pdf' inside the /data folder."
        )
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[ingest] Loaded {len(documents)} pages from {os.path.basename(pdf_path)}")
    return documents


def chunk_documents(documents):
    """Split pages into overlapping chunks suitable for embedding + retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] Split into {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


def build_and_save_index(chunks):
    """Embed all chunks and persist a FAISS index + a plain-text metadata backup."""
    print(f"[ingest] Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    print("[ingest] Generating embeddings and building FAISS index ...")
    vectorstore = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    vectorstore.save_local(config.INDEX_PATH)


    metadata = [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[ingest] FAISS index saved to:      {config.INDEX_PATH}")
    print(f"[ingest] Chunk metadata saved to:   {config.METADATA_PATH}")


def main():
    documents = load_pdf(config.PDF_PATH)
    chunks = chunk_documents(documents)
    build_and_save_index(chunks)
    print("\n✅ Ingestion complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
