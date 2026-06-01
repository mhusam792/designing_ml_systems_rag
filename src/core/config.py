from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

CHROMA_DB_DIR = ROOT_DIR / "data/processed/chroma_db"

LLM_MODEL = "llama3.2:3b"

TOP_K = 5

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

API_TITLE = "Book RAG API"

API_VERSION = "v1.0.0-beta2"
