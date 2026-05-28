from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

from utils import embeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
CHROMA_DB_DIR = "./chroma_db"


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('hierarchy_path')}]\n{d.page_content}" for d in docs
    )


def build_prompt(x: dict) -> str:
    return f"""
                You are an expert assistant.

                Answer ONLY using the context below.

                Context:
                {x['context']}

                Question:
                {x['question']}

                Answer:
            """.strip()


def build_rag_chain(
    chroma_db_dir: str = CHROMA_DB_DIR,
    llm_model: str = LLM_MODEL,
    top_k: int = 5,
):
    """
    Build a full RAG pipeline using Chroma + Ollama.

    Returns:
        LangChain Runnable (RAG chain)
    """

    vectorstore = Chroma(persist_directory=chroma_db_dir, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    format_docs_runnable = RunnableLambda(format_docs)

    llm = ChatOllama(model=llm_model, temperature=0)

    prompt_runnable = RunnableLambda(build_prompt)

    rag_chain = (
        {
            "context": retriever | format_docs_runnable,
            "question": RunnablePassthrough(),
        }
        | prompt_runnable
        | llm
        | StrOutputParser()
    )

    return rag_chain
