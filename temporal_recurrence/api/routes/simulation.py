"""API routes for running simulations and getting simulation presets."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from ..services.dsbm_service import run_live_simulation

router = APIRouter(prefix="/api", tags=["simulation"])


class SimulationRequest(BaseModel):
    num_nodes: int = Field(60, ge=15, le=150, description="Total nodes in the graph")
    num_communities: int = Field(3, ge=2, le=5, description="Number of community partitions")
    sequence_type: str = Field("recurrence", description="recurrence, permanent_drift, stationary, non_recurring")
    duration_a1: int = Field(25, ge=5, le=100)
    duration_b: int = Field(35, ge=5, le=100)
    duration_a2: int = Field(25, ge=5, le=100)
    target_density: float = Field(0.12, ge=0.02, le=0.5)
    lambda_a: float = Field(0.85, ge=0.0, le=0.99)
    lambda_b: float = Field(0.20, ge=0.0, le=0.99)
    multiplier_a: float = Field(3.5, ge=0.5, le=8.0)
    multiplier_b: float = Field(0.8, ge=0.1, le=5.0)
    seed: Optional[int] = Field(42, description="Random seed")


@router.post("/simulate")
def simulate_dsbm(req: SimulationRequest) -> Dict[str, Any]:
    try:
        data = run_live_simulation(
            num_nodes=req.num_nodes,
            num_communities=req.num_communities,
            sequence_type=req.sequence_type,
            duration_a1=req.duration_a1,
            duration_b=req.duration_b,
            duration_a2=req.duration_a2,
            target_density=req.target_density,
            lambda_a=req.lambda_a,
            lambda_b=req.lambda_b,
            multiplier_a=req.multiplier_a,
            multiplier_b=req.multiplier_b,
            seed=req.seed
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presets")
def get_presets() -> Dict[str, Any]:
    return {
        "presets": [
            {
                "id": "recurrence_standard",
                "name": "Standard Recurrence (A -> B -> A)",
                "description": "Tests if models retrieve historical Regime A when it recurs after a volatile Regime B.",
                "sequence_type": "recurrence",
                "num_nodes": 60,
                "duration_a1": 25,
                "duration_b": 35,
                "duration_a2": 25,
                "lambda_a": 0.85,
                "lambda_b": 0.20,
                "multiplier_a": 3.5,
                "multiplier_b": 0.8
            },
            {
                "id": "long_intervening",
                "name": "Long Intervening B-Phase (TB = 60)",
                "description": "Extreme recency pressure test: 60 timesteps in B before returning to A.",
                "sequence_type": "recurrence",
                "num_nodes": 60,
                "duration_a1": 20,
                "duration_b": 60,
                "duration_a2": 25,
                "lambda_a": 0.90,
                "lambda_b": 0.15,
                "multiplier_a": 4.0,
                "multiplier_b": 0.5
            },
            {
                "id": "permanent_drift",
                "name": "Permanent Drift Control (A -> B)",
                "description": "Baseline control evaluating adaptation when regime permanently switches.",
                "sequence_type": "permanent_drift",
                "num_nodes": 60,
                "duration_a1": 35,
                "duration_b": 45,
                "duration_a2": 0,
                "lambda_a": 0.85,
                "lambda_b": 0.20,
                "multiplier_a": 3.5,
                "multiplier_b": 0.8
            },
            {
                "id": "non_recurring",
                "name": "Non-Recurring Shift (A -> B -> C)",
                "description": "Validates that historical A retrieval is specific and doesn't fire for novel Regime C.",
                "sequence_type": "non_recurring",
                "num_nodes": 60,
                "duration_a1": 25,
                "duration_b": 35,
                "duration_a2": 25,
                "lambda_a": 0.85,
                "lambda_b": 0.20,
                "multiplier_a": 3.5,
                "multiplier_b": 0.8
            }
        ]
    }
