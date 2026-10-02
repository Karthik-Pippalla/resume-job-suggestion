from fastapi import APIRouter, Request

from app.schemas import Job
from app.services.matcher import JobIndex

router = APIRouter(prefix="/api/v1", tags=["jobs"])


@router.get("/jobs", response_model=list[Job])
def list_jobs(request: Request) -> list[Job]:
    index: JobIndex = request.app.state.index
    return index.jobs
