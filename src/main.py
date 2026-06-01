import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

from src.services.qa_service import QAService
from src.api.routes import router
from src.core.config import CHROMA_DB_DIR, LLM_MODEL, TOP_K
from src.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.qa_service = QAService(
        chroma_dir=str(CHROMA_DB_DIR),
        llm_model=LLM_MODEL,
        top_k=TOP_K,
    )

    yield

    print("Shutdown")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = round(
        time.perf_counter() - start_time,
        3,
    )

    logger.info(
        f"{request.method} "
        f"{request.url.path} "
        f"status={response.status_code} "
        f"duration={duration}s"
    )

    return response


app.include_router(router)
