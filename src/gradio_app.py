import gradio as gr
import requests
from src.core.config import GRADIO_API_URL

API_URL = GRADIO_API_URL


def ask_question(question: str):

    response = requests.post(
        API_URL,
        json={"question": question},
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    answer = result["answer"]

    sources = "\n\n".join(f"""
                            Hierarchy: {source['hierarchy']}
                            Chapter: {source['chapter']}
                            Section: {source['section']}
                            Page: {source['page']}
                            """.strip() for source in result["sources"])

    chunks = ("\n\n" + "=" * 80 + "\n\n").join(result["chunks"])

    return answer, sources, chunks


with gr.Blocks(title="Designing ML Systems RAG") as demo:

    gr.Markdown("""
# 📚 Designing Machine Learning Systems RAG

Ask questions about the book and inspect the retrieved context.
""")

    with gr.Row():
        question = gr.Textbox(
            label="Question",
            placeholder="Ask a question about the book...",
            lines=2,
            scale=8,
        )

        submit_btn = gr.Button(
            "Submit",
            scale=1,
        )

    answer = gr.Textbox(
        label="Answer",
        lines=10,
    )

    sources = gr.Textbox(
        label="Retrieved Sources",
        lines=10,
    )

    chunks = gr.Code(
        label="Retrieved Chunks",
        language="markdown",
    )

    submit_btn.click(
        fn=ask_question,
        inputs=question,
        outputs=[
            answer,
            sources,
            chunks,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
