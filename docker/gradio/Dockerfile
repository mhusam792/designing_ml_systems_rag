FROM rag-base

WORKDIR /app

COPY src ./src

EXPOSE 7860

CMD ["uv", "run", "python", "-m", "src.gradio_app"]