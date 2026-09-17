"""API routes for exploring experiment phases, verdicts, benchmarks, and figures."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Dict, Any, List

from ..services.results_service import get_available_phases, get_phase_details, get_figure_path

router = APIRouter(prefix="/api", tags=["results"])


@router.get("/phases")
def list_phases() -> Dict[str, Any]:
    return {"phases": get_available_phases()}


@router.get("/phases/{phase_id}")
def phase_info(phase_id: str) -> Dict[str, Any]:
    details = get_phase_details(phase_id)
    if "error" in details:
        raise HTTPException(status_code=404, detail=details["error"])
    return details


@router.get("/figures/{phase_id}/{filename}")
def serve_figure(phase_id: str, filename: str):
    fig_path = get_figure_path(phase_id, filename)
    if not fig_path or not fig_path.exists():
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(str(fig_path), media_type="image/png")
