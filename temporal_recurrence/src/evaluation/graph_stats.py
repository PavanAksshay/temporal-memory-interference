"""Graph statistics calculation across time and regimes."""
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from ..generator.dsbm import DynamicGraphSequence


def compute_timestep_stats(G_t: np.ndarray, G_prev: np.ndarray | None, community_assignments: np.ndarray) -> Dict[str, float]:
    """Compute structural graph statistics for a single snapshot."""
    N = G_t.shape[0]
    triu_idx = np.triu_indices(N, k=1)
    edges = G_t[triu_idx]
    num_edges = int(np.sum(edges))
    total_pairs = len(edges)
    edge_density = float(num_edges / total_pairs) if total_pairs > 0 else 0.0

    degrees = np.sum(G_t, axis=1)
    mean_degree = float(np.mean(degrees))
    degree_std = float(np.std(degrees))

    c = community_assignments
    same_comm_mask = (c[triu_idx[0]] == c[triu_idx[1]])
    within_edges = edges[same_comm_mask]
    between_edges = edges[~same_comm_mask]

    within_density = float(np.mean(within_edges)) if len(within_edges) > 0 else 0.0
    between_density = float(np.mean(between_edges)) if len(between_edges) > 0 else 0.0

    if G_prev is not None:
        prev_edges = G_prev[triu_idx]
        intersection = np.sum((edges == 1) & (prev_edges == 1))
        union = np.sum((edges == 1) | (prev_edges == 1))
        jaccard_overlap = float(intersection / union) if union > 0 else 0.0
        # Edge persistence: fraction of previous edges that survived
        persistence = float(intersection / np.sum(prev_edges)) if np.sum(prev_edges) > 0 else 0.0
    else:
        jaccard_overlap = np.nan
        persistence = np.nan

    return {
        "num_edges": num_edges,
        "edge_density": edge_density,
        "mean_degree": mean_degree,
        "degree_std": degree_std,
        "within_comm_density": within_density,
        "between_comm_density": between_density,
        "jaccard_overlap": jaccard_overlap,
        "edge_persistence": persistence
    }


def compute_sequence_graph_stats(seq: DynamicGraphSequence) -> pd.DataFrame:
    """Calculate temporal statistics for the entire sequence."""
    records = []
    T = seq.total_timesteps
    c = seq.community_assignments

    for t in range(T):
        rec = seq.scheduler.get_record(t)
        G_t = seq.get_snapshot(t)
        G_prev = seq.get_snapshot(t - 1) if t > 0 else None
        stats = compute_timestep_stats(G_t, G_prev, c)
        
        row = {
            "time": t,
            "regime": rec.regime,
            "episode_index": rec.episode_index,
            "time_since_transition": rec.time_since_transition,
            **stats
        }
        records.append(row)

    df = pd.DataFrame(records)
    return df


def aggregate_regime_stats(stats_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute mean and std metrics grouped by regime."""
    grouped = stats_df.groupby("regime")
    summary = {}
    for regime_name, group in grouped:
        summary[regime_name] = {
            "mean_edge_density": float(group["edge_density"].mean()),
            "std_edge_density": float(group["edge_density"].std()),
            "mean_degree": float(group["mean_degree"].mean()),
            "within_comm_density": float(group["within_comm_density"].mean()),
            "between_comm_density": float(group["between_comm_density"].mean()),
            "mean_jaccard_overlap": float(group["jaccard_overlap"].dropna().mean()),
            "mean_persistence": float(group["edge_persistence"].dropna().mean())
        }
    return summary
