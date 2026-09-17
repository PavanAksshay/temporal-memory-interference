"""FastAPI application for Temporal Graph Recurrence Visualizer & Dashboard."""
import sys
from pathlib import Path

# Add root directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.simulation import router as sim_router
from api.routes.results import router as res_router

app = FastAPI(
    title="Temporal Graph Recurrence API",
    description="Interactive backend for dynamic DSBM simulation, baseline forecasting evaluation, and experiment benchmark inspection.",
    version="1.0.0"
)

# Enable CORS for local dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sim_router)
app.include_router(res_router)


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Temporal Graph Recurrence Research API",
        "version": "1.0.0",
        "endpoints": ["/api/simulate", "/api/presets", "/api/phases", "/api/phases/{id}"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
