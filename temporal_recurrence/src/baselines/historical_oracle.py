"""Historical, True Regime, and Random-History Oracle predictors."""
from typing import List, Optional
import numpy as np

from ..evaluation.prediction import verify_no_future_leakage


class HistoricalOraclePredictor:
    """
    Oracle 3: Previous-A History Predictor.
    Combines current state G_t with historical observations from previous A episode.
    Answers: 'Is useful historical information actually available?'
    """

    def __init__(
        self,
        variant: str = "historical_summary",  # "exact_snapshot" or "historical_summary"
        history_weight: float = 1.0,
        cn_weight: float = 0.3
    ):
        self.variant = variant
        self.history_weight = history_weight
        self.cn_weight = cn_weight

    def predict_pairs(
        self,
        G_t: np.ndarray,
        historical_A_snapshots: List[np.ndarray],
        pairs: np.ndarray,
        current_time: int,
        accessed_times: List[int]
    ) -> np.ndarray:
        """
        G_t: Current graph at timestep t.
        historical_A_snapshots: Graph snapshots strictly from previous A regime.
        pairs: Node pairs (u, v) to evaluate.
        current_time: Active timestep t.
        accessed_times: Timestamps accessed to form historical_A_snapshots.
        """
        verify_no_future_leakage(current_time, accessed_times)

        if len(pairs) == 0:
            return np.empty((0,), dtype=np.float64)

        u = pairs[:, 0]
        v = pairs[:, 1]
        current_edges = G_t[u, v].astype(np.float64)

        if len(historical_A_snapshots) == 0:
            cn_scores = np.sum(G_t[u] * G_t[v], axis=1).astype(np.float64)
            max_cn = np.max(cn_scores) if len(cn_scores) > 0 and np.max(cn_scores) > 0 else 1.0
            return current_edges + self.cn_weight * (cn_scores / max_cn)

        if self.variant == "exact_snapshot":
            H = historical_A_snapshots[-1].astype(np.float64)
        else:
            H = np.mean([snap.astype(np.float64) for snap in historical_A_snapshots], axis=0)

        hist_edges = H[u, v]
        blended = G_t.astype(np.float64) + self.history_weight * H
        cn_scores = np.sum(blended[u] * blended[v], axis=1)
        max_cn = np.max(cn_scores) if len(cn_scores) > 0 and np.max(cn_scores) > 0 else 1.0

        scores = current_edges + self.history_weight * hist_edges + self.cn_weight * (cn_scores / max_cn)
        return scores


class TrueRegimeOraclePredictor:
    """
    Oracle 4: True Regime Oracle.
    Upper-bound diagnostic that has access to the ground-truth active regime affinity matrix W^{(Z_t)}.
    """

    def __init__(self, affinity_weight: float = 1.5, cn_weight: float = 0.3):
        self.affinity_weight = affinity_weight
        self.cn_weight = cn_weight

    def predict_pairs(
        self,
        G_t: np.ndarray,
        true_affinity_matrix: np.ndarray,
        pairs: np.ndarray
    ) -> np.ndarray:
        if len(pairs) == 0:
            return np.empty((0,), dtype=np.float64)

        u = pairs[:, 0]
        v = pairs[:, 1]

        current_edges = G_t[u, v].astype(np.float64)
        true_affinities = true_affinity_matrix[u, v]

        cn_scores = np.sum(true_affinity_matrix[u] * true_affinity_matrix[v], axis=1)
        max_cn = np.max(cn_scores) if len(cn_scores) > 0 and np.max(cn_scores) > 0 else 1.0

        return current_edges + self.affinity_weight * true_affinities + self.cn_weight * (cn_scores / max_cn)


class RandomHistoryControlPredictor:
    """
    Negative Control: Combines G_t with a randomly selected historical snapshot from permitted past.
    Verifies that oracle advantage is truly due to regime recurrence, not merely adding more historical data.
    """

    def __init__(self, history_weight: float = 1.0, cn_weight: float = 0.3):
        self.history_weight = history_weight
        self.cn_weight = cn_weight

    def predict_pairs(
        self,
        G_t: np.ndarray,
        random_past_snapshots: List[np.ndarray],
        pairs: np.ndarray,
        current_time: int,
        accessed_times: List[int]
    ) -> np.ndarray:
        verify_no_future_leakage(current_time, accessed_times)

        if len(pairs) == 0:
            return np.empty((0,), dtype=np.float64)

        u = pairs[:, 0]
        v = pairs[:, 1]
        current_edges = G_t[u, v].astype(np.float64)

        if len(random_past_snapshots) == 0:
            return current_edges

        H = np.mean([snap.astype(np.float64) for snap in random_past_snapshots], axis=0)
        hist_edges = H[u, v]

        blended = G_t.astype(np.float64) + self.history_weight * H
        cn_scores = np.sum(blended[u] * blended[v], axis=1)
        max_cn = np.max(cn_scores) if len(cn_scores) > 0 and np.max(cn_scores) > 0 else 1.0

        return current_edges + self.history_weight * hist_edges + self.cn_weight * (cn_scores / max_cn)
