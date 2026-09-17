"""Recent-history baseline predictor using sliding window history."""
from typing import List
import numpy as np


class RecentHistoryPredictor:
    """
    Predicts Y_{e, t+1} using history window G_{t-h+1:t}.
    Represents a recency-oriented learner.
    """

    def __init__(self, window_size: int = 5, decay_gamma: float = 0.8, cn_weight: float = 0.5):
        self.window_size = window_size
        self.decay_gamma = decay_gamma
        self.cn_weight = cn_weight

    def predict_pairs(self, history_snapshots: List[np.ndarray], pairs: np.ndarray) -> np.ndarray:
        """
        history_snapshots: List of G_tau for tau in [max(0, t - window_size + 1), t]
        Most recent snapshot is history_snapshots[-1].
        """
        if len(pairs) == 0:
            return np.empty((0,), dtype=np.float64)

        u = pairs[:, 0]
        v = pairs[:, 1]
        k = len(history_snapshots)
        if k == 0:
            return np.zeros(len(pairs), dtype=np.float64)

        # Weighted moving average of edge states
        weights = [self.decay_gamma ** (k - 1 - i) for i in range(k)]
        total_weight = sum(weights)

        weighted_edges = np.zeros(len(pairs), dtype=np.float64)
        for i, G in enumerate(history_snapshots):
            weighted_edges += weights[i] * G[u, v].astype(np.float64)
        weighted_edges /= total_weight

        # Weighted common neighbors
        # Use average recent snapshot
        avg_G = np.zeros_like(history_snapshots[0], dtype=np.float64)
        for i, G in enumerate(history_snapshots):
            avg_G += weights[i] * G.astype(np.float64)
        avg_G /= total_weight

        cn_scores = np.sum(avg_G[u] * avg_G[v], axis=1)
        max_cn = np.max(cn_scores) if len(cn_scores) > 0 and np.max(cn_scores) > 0 else 1.0

        scores = weighted_edges + self.cn_weight * (cn_scores / max_cn)
        return scores
