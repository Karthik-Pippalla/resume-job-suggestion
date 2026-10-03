from pydantic import BaseModel, Field


class Fund(BaseModel):
    id: str
    name: str
    ticker: str
    sector: str
    risk: str
    focus_skills: list[str]
    related_skills: list[str]
    description: str


class Recommendation(BaseModel):
    fund: Fund
    score: float = Field(description="Weighted match score from 0 to 1")
    focus_coverage: float
    related_coverage: float
    text_similarity: float
    matched_skills: list[str]
    missing_skills: list[str]


class RecommendationResponse(BaseModel):
    extracted_skills: list[str]
    resume_years: float | None = Field(
        description="Years of work read from the resume. Null when none could be read. Not used to pick a fund."
    )
    recommendations: list[Recommendation]


class HealthResponse(BaseModel):
    status: str
