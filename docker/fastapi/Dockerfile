FROM python:3.11-slim

WORKDIR /app

COPY src ./src
COPY data ./data

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]