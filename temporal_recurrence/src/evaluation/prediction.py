"""Link prediction target extraction, edge sampling, and scoring metrics."""
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sample_evaluation_edges(
    G_target: np.ndarray,
    negative_ratio: float = 1.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample positive and negative node pairs from target graph G_{t+1}.
    Returns:
        pairs: np.ndarray of shape (M, 2) containing node index pairs (u, v) with u < v.
        labels: np.ndarray of shape (M,) with 1 for present edges, 0 for absent edges.
    """
    rng = np.random.default_rng(seed)
    N = G_target.shape[0]
    triu_i, triu_j = np.triu_indices(N, k=1)
    all_states = G_target[triu_i, triu_j]

    pos_mask = (all_states == 1)
    neg_mask = (all_states == 0)

    pos_u = triu_i[pos_mask]
    pos_v = triu_j[pos_mask]
    num_pos = len(pos_u)

    if num_pos == 0:
        # Edge case: empty target graph
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int)

    neg_u_all = triu_i[neg_mask]
    neg_v_all = triu_j[neg_mask]
    num_neg_target = int(num_pos * negative_ratio)
    num_neg_target = min(num_neg_target, len(neg_u_all))

    neg_indices = rng.choice(len(neg_u_all), size=num_neg_target, replace=False)
    neg_u = neg_u_all[neg_indices]
    neg_v = neg_v_all[neg_indices]

    pairs_u = np.concatenate([pos_u, neg_u])
    pairs_v = np.concatenate([pos_v, neg_v])
    labels = np.concatenate([np.ones(num_pos, dtype=np.int32), np.zeros(num_neg_target, dtype=np.int32)])

    # Shuffle evaluation pairs
    perm = rng.permutation(len(labels))
    pairs = np.column_stack([pairs_u[perm], pairs_v[perm]])
    labels = labels[perm]

    return pairs, labels


def compute_prediction_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate Average Precision (AP) and Area Under ROC Curve (AUC)."""
    if len(labels) == 0 or np.sum(labels) == 0 or np.sum(labels) == len(labels):
        return {"ap": 0.0, "auc": 0.5}

    ap = float(average_precision_score(labels, scores))
    auc = float(roc_auc_score(labels, scores))
    return {"ap": ap, "auc": auc}


def verify_no_future_leakage(current_time: int, accessed_times: list[int]) -> None:
    """Strict assertion that features only read timesteps <= current_time."""
    for t in accessed_times:
        if t > current_time:
            raise ValueError(f"CRITICAL DATA LEAKAGE: Evaluator at t={current_time} accessed future graph t={t}!")
