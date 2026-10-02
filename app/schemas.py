from pydantic import BaseModel, Field


class Job(BaseModel):
    id: str
    title: str
    company: str
    location: str
    seniority: str
    min_years: float = 0
    max_years: float | None = None
    required_skills: list[str]
    preferred_skills: list[str]
    description: str


class Recommendation(BaseModel):
    job: Job
    score: float = Field(description="Weighted match score from 0 to 1")
    required_coverage: float
    preferred_coverage: float
    text_similarity: float
    experience_fit: float
    matched_skills: list[str]
    missing_skills: list[str]


class RecommendationResponse(BaseModel):
    extracted_skills: list[str]
    resume_years: float | None = Field(
        description="Years of experience read from the resume. Null when none could be read."
    )
    recommendations: list[Recommendation]


class HealthResponse(BaseModel):
    status: str
