from langchain_huggingface import HuggingFaceEmbeddings

from src.core.config import EMBEDDING_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
