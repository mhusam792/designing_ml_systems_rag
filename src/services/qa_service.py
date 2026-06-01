import time

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from src.utils import embeddings
from src.core.logging import logger


class QAService:

    def __init__(
        self,
        chroma_dir: str,
        llm_model: str,
        top_k: int = 5,
    ):
        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings,
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        self.llm = ChatOllama(
            model=llm_model,
            temperature=0,
        )

    def ask(self, question: str):

        start_time = time.perf_counter()

        try:

            docs = self.retriever.invoke(question)
            retrieved_docs_count = len(docs)

            context = "\n\n---\n\n".join(
                f"[{d.metadata.get('hierarchy_path')}]\n{d.page_content}" for d in docs
            )

            prompt = f"""
                        You are an expert assistant.

                        Answer ONLY using the context below.

                        Context:
                        {context}

                        Question:
                        {question}

                        Answer:
                    """.strip()

            answer = self.llm.invoke(prompt).content

            duration = round(time.perf_counter() - start_time, 3)

            logger.info(
                f"question='{question}' | "
                f"chunks={retrieved_docs_count} | "
                f"duration={duration}s"
            )

            return {
                "answer": answer,
                "sources": [
                    {
                        "chapter": d.metadata.get("chapter_title"),
                        "section": d.metadata.get("section_title"),
                        "page": d.metadata.get("page_label"),
                        "hierarchy": d.metadata.get("hierarchy_path"),
                    }
                    for d in docs
                ],
                "chunks": [doc.page_content for doc in docs],
            }

        except Exception as e:

            logger.exception(f"query_failed='{question}'")

            raise
