from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.jobs import router as jobs_router
from app.api.recommendations import router as recommendations_router
from app.schemas import HealthResponse
from app.services.matcher import JobIndex

HOME_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index = JobIndex.from_data_dir()
    yield


app = FastAPI(
    title="Resume Job Suggestion API",
    summary="Rank a seeded job catalog against an uploaded resume.",
    lifespan=lifespan,
)
app.include_router(recommendations_router)
app.include_router(jobs_router)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(HOME_PAGE)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
