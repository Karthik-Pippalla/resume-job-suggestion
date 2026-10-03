from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.investments import router as investments_router
from app.api.recommendations import router as recommendations_router
from app.schemas import HealthResponse
from app.services.matcher import FundIndex

HOME_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index = FundIndex.from_data_dir()
    yield


app = FastAPI(
    title="Resume Investment Suggestion API",
    summary="Rank a seeded sample fund catalog against an uploaded resume.",
    lifespan=lifespan,
)
app.include_router(recommendations_router)
app.include_router(investments_router)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(HOME_PAGE)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
