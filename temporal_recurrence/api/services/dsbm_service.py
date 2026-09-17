"""Service for running live DSBM simulations and evaluating baselines."""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Ensure root temporal_recurrence is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.dsbm import DynamicSBMGenerator, DynamicGraphSequence
from src.generator.regimes import RegimeConfig
from src.baselines.current_only import CurrentOnlyPredictor
from src.baselines.recent_history import RecentHistoryPredictor
from src.baselines.historical_oracle import HistoricalOraclePredictor
from src.evaluation.prediction import sample_evaluation_edges, compute_prediction_metrics


def run_live_simulation(
    num_nodes: int = 60,
    num_communities: int = 3,
    sequence_type: str = "recurrence",
    duration_a1: int = 25,
    duration_b: int = 35,
    duration_a2: int = 25,
    target_density: float = 0.12,
    lambda_a: float = 0.85,
    lambda_b: float = 0.20,
    multiplier_a: float = 3.5,
    multiplier_b: float = 0.8,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Runs a fast simulation suited for interactive visualization and returns
    node positions, community assignments, frames (edge lists), and time-series metrics.
    """
    if seed is not None:
        np.random.seed(seed)

    # Build sequence specification
    if sequence_type == "recurrence":
        sequence_spec = [("A", duration_a1), ("B", duration_b), ("A", duration_a2)]
    elif sequence_type == "permanent_drift":
        sequence_spec = [("A", duration_a1), ("B", duration_b)]
    elif sequence_type == "stationary":
        sequence_spec = [("A", duration_a1 + duration_b + duration_a2)]
    elif sequence_type == "non_recurring":
        sequence_spec = [("A", duration_a1), ("B", duration_b), ("C", duration_a2)]
    else:
        sequence_spec = [("A", duration_a1), ("B", duration_b), ("A", duration_a2)]

    regime_configs = {
        "A": RegimeConfig(
            name="A",
            target_density=target_density,
            persistence=lambda_a,
            within_comm_multiplier=multiplier_a
        ),
        "B": RegimeConfig(
            name="B",
            target_density=target_density,
            persistence=lambda_b,
            within_comm_multiplier=multiplier_b
        ),
        "C": RegimeConfig(
            name="C",
            target_density=target_density,
            persistence=0.50,
            within_comm_multiplier=1.5
        )
    }

    generator = DynamicSBMGenerator(
        num_nodes=num_nodes,
        num_communities=num_communities,
        mode="temporal_plus_structure",
        regime_configs=regime_configs
    )

    sequence: DynamicGraphSequence = generator.generate(sequence_spec=sequence_spec, seed=seed)
    T, N, _ = sequence.snapshots.shape

    # Precompute 2D community cluster anchor positions for visualization
    nodes = []
    community_centers = []
    angle_step = 2 * np.pi / num_communities
    radius = 160.0

    for c in range(num_communities):
        angle = c * angle_step - (np.pi / 2)
        cx = 300 + radius * np.cos(angle)
        cy = 300 + radius * np.sin(angle)
        community_centers.append((cx, cy))

    for i in range(N):
        comm = int(sequence.community_assignments[i])
        cx, cy = community_centers[comm]
        # Cluster offset
        jitter_r = np.random.uniform(10, 65)
        jitter_theta = np.random.uniform(0, 2 * np.pi)
        x = float(cx + jitter_r * np.cos(jitter_theta))
        y = float(cy + jitter_r * np.sin(jitter_theta))
        nodes.append({
            "id": i,
            "community": comm,
            "x": round(x, 2),
            "y": round(y, 2),
            "cx": round(cx, 2),
            "cy": round(cy, 2)
        })

    # Predictors
    current_predictor = CurrentOnlyPredictor()
    recent_predictor = RecentHistoryPredictor(window_size=5, decay_gamma=0.8)
    oracle_predictor = HistoricalOraclePredictor(variant="historical_summary")

    timeline_records = []
    metrics_timeseries = []
    frames = []

    # Fast evaluation loop
    for t in range(T):
        record = sequence.scheduler.get_record(t)
        regime = record.regime
        G_t = sequence.snapshots[t]

        # Edge list for frame
        u_idx, v_idx = np.where(np.triu(G_t, k=1))
        edges = [{"source": int(u), "target": int(v)} for u, v in zip(u_idx, v_idx)]

        # Density and temporal overlap
        density = float(np.sum(G_t) / (N * (N - 1)))
        overlap = 0.0
        if t > 0:
            G_prev = sequence.snapshots[t - 1]
            intersection = np.logical_and(G_t, G_prev).sum()
            union = np.logical_or(G_t, G_prev).sum()
            overlap = float(intersection / union) if union > 0 else 0.0

        # Predictor scores if t < T - 1
        current_ap = 0.5
        recent_ap = 0.5
        oracle_ap = 0.5
        delta_old = 0.0
        delta_recent = 0.0

        if t < T - 1:
            G_next = sequence.snapshots[t + 1]
            pairs, labels = sample_evaluation_edges(
                G_target=G_next,
                negative_ratio=2.0,
                seed=seed + t if seed else 42
            )

            if len(labels) > 0 and labels.sum() > 0:
                # Current only
                curr_scores = current_predictor.predict_pairs(G_t, pairs)
                curr_m = compute_prediction_metrics(curr_scores, labels)
                current_ap = float(curr_m["ap"])

                # Recent history
                h_start = max(0, t - 4)
                recent_history = [sequence.snapshots[tau] for tau in range(h_start, t + 1)]
                rec_scores = recent_predictor.predict_pairs(recent_history, pairs)
                rec_m = compute_prediction_metrics(rec_scores, labels)
                recent_ap = float(rec_m["ap"])

                # Historical oracle
                prev_episodes = sequence.scheduler.get_previous_episodes_for_regime(regime, before_time=t)
                hist_snapshots = []
                accessed_times = []
                for ep in prev_episodes:
                    ep_times = list(range(int(ep["start"]), int(ep["end"]) + 1))
                    accessed_times.extend(ep_times)
                    hist_snapshots.extend([sequence.snapshots[tau] for tau in ep_times])
                
                if hist_snapshots:
                    orc_scores = oracle_predictor.predict_pairs(
                        G_t=G_t,
                        historical_A_snapshots=hist_snapshots,
                        pairs=pairs,
                        current_time=t,
                        accessed_times=accessed_times
                    )
                    orc_m = compute_prediction_metrics(orc_scores, labels)
                    oracle_ap = float(orc_m["ap"])
                    delta_old = oracle_ap - current_ap
                else:
                    oracle_ap = current_ap
                    delta_old = 0.0

                delta_recent = recent_ap - current_ap

        timeline_records.append({
            "t": t,
            "regime": regime,
            "episode": record.episode_index,
            "time_since_transition": record.time_since_transition
        })

        metrics_timeseries.append({
            "t": t,
            "regime": regime,
            "density": round(density, 4),
            "overlap": round(overlap, 4),
            "current_ap": round(current_ap, 4),
            "recent_ap": round(recent_ap, 4),
            "oracle_ap": round(oracle_ap, 4),
            "delta_old": round(delta_old, 4),
            "delta_recent": round(delta_recent, 4),
            "num_edges": len(edges)
        })

        frames.append({
            "t": t,
            "regime": regime,
            "num_edges": len(edges),
            "edges": edges
        })

    # Summary statistics
    episodes_meta = sequence.scheduler.episodes
    total_edges = sum(f["num_edges"] for f in frames)

    # Compute a 2D Attention / Retrieval Matrix across key time slices
    attention_matrix = []
    step_gap = max(1, T // 25)
    query_steps = list(range(0, T, step_gap))
    if query_steps[-1] != T - 1:
        query_steps.append(T - 1)

    for q_t in query_steps:
        q_regime = timeline_records[q_t]["regime"]
        row = {"query_t": q_t, "regime": q_regime, "weights": []}
        for k_t in query_steps:
            if k_t > q_t:
                weight = 0.0
            else:
                k_regime = timeline_records[k_t]["regime"]
                lag = q_t - k_t
                recency_comp = np.exp(-0.07 * lag)
                match_boost = 0.65 if (k_regime == q_regime and lag > 5) else 0.05
                weight = float(0.4 * recency_comp + 0.6 * match_boost)
            row["weights"].append(round(weight, 4))
        # Normalize weights
        total_w = sum(row["weights"])
        if total_w > 0:
            row["weights"] = [round(w / total_w, 4) for w in row["weights"]]
        attention_matrix.append(row)

    return {
        "metadata": {
            "num_nodes": N,
            "num_communities": num_communities,
            "total_timesteps": T,
            "sequence_type": sequence_type,
            "episodes": episodes_meta,
            "total_edges_sampled": total_edges,
            "regime_configs": {
                k: {
                    "density": v.target_density,
                    "lambda": v.persistence,
                    "multiplier": v.within_comm_multiplier,
                    "description": f"Regime {k}"
                }
                for k, v in regime_configs.items()
            }
        },
        "nodes": nodes,
        "frames": frames,
        "timeline": timeline_records,
        "metrics": metrics_timeseries,
        "attention_slices": {
            "query_steps": query_steps,
            "matrix": attention_matrix
        }
    }
