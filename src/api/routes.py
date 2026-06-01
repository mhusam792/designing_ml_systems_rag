from fastapi import APIRouter, Request, HTTPException

from src.api.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.post("/query", response_model=QueryResponse)
def query(request_body: QueryRequest, request: Request):

    try:
        qa_service = request.app.state.qa_service

        return qa_service.ask(request_body.question)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
