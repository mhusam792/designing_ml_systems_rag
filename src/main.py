from chunking import build_chunks_with_metadata
from vectordb import build_vector_database
from rag_chain import build_rag_chain
from pdf_structure import build_pdf_structure

import json

PDF_PATH = "data/raw/01_Designing_Machine_Learning_Systems.pdf"
STRUCTURE_JSON = "data/processed/book_structure.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_DIR = "./chroma_db"
QUESTIONS_ANSWERS_PATH = "data/raw/qa.jsonl"

LLM_MODEL = "llama3.2:3b"


if __name__ == "__main__":

    df = build_pdf_structure(PDF_PATH, STRUCTURE_JSON)

    chunks = build_chunks_with_metadata(
        pdf_path=PDF_PATH,
        structure_json=STRUCTURE_JSON,
    )

    print(f"Total chunks: {len(chunks)}")

    vectorstore = build_vector_database(chunks=chunks, persist_directory=CHROMA_DB_DIR)

    rag_chain = build_rag_chain()

    data = []
    with open(QUESTIONS_ANSWERS_PATH) as f:
        for line in f:
            data.append(json.loads(line))

    for q in data:
        print(q["question"])
        print()

        response = rag_chain.invoke(q["question"])
        print(response)

        print("#" * 50)
        print()
