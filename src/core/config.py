from pathlib import Path
from dotenv import load_dotenv

import os

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]

CHROMA_DB_DIR = ROOT_DIR / "data/processed/chroma_db"

API_TITLE = os.getenv(
    "API_TITLE",
    "Book RAG API",
)

API_VERSION = os.getenv(
    "API_VERSION",
    "v1.0.0-beta2",
)

API_HOST = os.getenv(
    "API_HOST",
    "0.0.0.0",
)

API_PORT = int(
    os.getenv(
        "API_PORT",
        "8000",
    )
)


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:3b",
)


TOP_K = int(
    os.getenv(
        "TOP_K",
        "5",
    )
)


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


GRADIO_HOST = os.getenv(
    "GRADIO_HOST",
    "0.0.0.0",
)

GRADIO_PORT = int(
    os.getenv(
        "GRADIO_PORT",
        "7860",
    )
)

GRADIO_API_URL = os.getenv(
    "GRADIO_API_URL",
    "http://127.0.0.1:8000/query",
)