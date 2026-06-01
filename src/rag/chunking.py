from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.metadata import load_structure_dataframe, enrich_chunks


def build_chunks_with_metadata(
    pdf_path: str,
    structure_json: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Load a PDF, split it into chunks, and enrich the chunks
    with chapter/section metadata.

    Args:
        pdf_path (str): Path to the PDF file.
        structure_json (str): Path to the structure JSON file.
        chunk_size (int): Chunk size.
        chunk_overlap (int): Chunk overlap.

    Returns:
        list[Document]: Enriched document chunks.
    """

    documents = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(documents)

    structure_df = load_structure_dataframe(structure_json)

    return enrich_chunks(chunks, structure_df)
