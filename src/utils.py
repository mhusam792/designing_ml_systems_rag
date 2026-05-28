from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
