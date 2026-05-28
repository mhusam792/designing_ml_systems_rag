from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_DIR = "./chroma_db"


def build_vector_database(
    chunks: list[Document],
    embedding_model: str = EMBEDDING_MODEL,
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
        tuple[HuggingFaceEmbeddings, Chroma]: Embedding model and persisted vector store.
    """

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)  # Free & local

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return embeddings, vectorstore
