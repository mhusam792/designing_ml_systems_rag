FROM python:3.11-slim

WORKDIR /app

COPY src ./src
COPY data ./data

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]