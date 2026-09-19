import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_PATH = os.path.join(DATA_DIR, "Ebook-Agentic-AI.pdf")

INDEX_DIR = os.path.join(BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index")
METADATA_PATH = os.path.join(INDEX_DIR, "chunks_metadata.pkl")

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.1

# Retrieval
TOP_K = 4

SIMILARITY_SCORE_THRESHOLD = 0.45
