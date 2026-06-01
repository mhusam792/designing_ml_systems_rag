from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.utils import embeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_DIR = "data/processed/chroma_db"


def build_vector_database(
    chunks: list[Document],
    persist_directory: str = CHROMA_DB_DIR,
) -> Chroma:
    """
    Build a vector database from document chunks.

    Pipeline:
        1. Initialize embedding model.
        2. Create Chroma vector store.
        3. Persist vectors to disk.

    Args:
        chunks (list[Document]): Document chunks.
        embedding_model (str): Embedding model name.
        persist_directory (str): Chroma database directory.

    Returns:
        Chroma: persisted vector store.
    """

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return vectorstore
