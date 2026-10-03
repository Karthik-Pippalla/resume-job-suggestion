from fastapi import APIRouter, Request

from app.schemas import Fund
from app.services.matcher import FundIndex

router = APIRouter(prefix="/api/v1", tags=["investments"])


@router.get("/investments", response_model=list[Fund])
def list_investments(request: Request) -> list[Fund]:
    index: FundIndex = request.app.state.index
    return index.funds
