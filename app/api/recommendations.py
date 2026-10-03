from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.schemas import RecommendationResponse
from app.services.llm_extract import OpenAIExtractError, extract_with_openai
from app.services.matcher import FundIndex
from app.services.parser import ResumeParseError, parse_resume

router = APIRouter(prefix="/api/v1", tags=["recommendations"])


@router.post("/recommendations", response_model=RecommendationResponse)
async def recommend(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
    openai_api_key: str | None = Form(default=None),
) -> RecommendationResponse:
    content = await file.read()
    try:
        text = parse_resume(file.filename, content)
    except ResumeParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    index: FundIndex = request.app.state.index
    skills = None
    years = None
    use_given_years = False
    if openai_api_key and openai_api_key.strip():
        try:
            skills, years = await extract_with_openai(
                text,
                index.extractor.lexicon,
                openai_api_key,
            )
        except OpenAIExtractError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        use_given_years = True

    return index.rank(
        text,
        top_k=top_k,
        skills=skills,
        years=years,
        use_given_years=use_given_years,
    )
