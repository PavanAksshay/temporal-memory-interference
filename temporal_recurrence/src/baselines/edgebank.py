"""
EdgeBank Exact Historical Edge Memorization Baseline.
Maintains a memory of observed edges over time to predict link recurrence.
"""
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from ..evaluation.prediction import verify_no_future_leakage


class EdgeBankPredictor:
    """
    EdgeBank exact historical edge memory baseline.
    Scores candidate node pairs (u, v) based on whether they appeared in
    the historical observation window.
    
    Modes:
    - 'unbounded' / 'all_history': Remembers all observed edges from t=0 up to t-1.
    - 'bounded': Remembers edges observed strictly within a specified historical window [t_start, t_end].
    """

    def __init__(
        self,
        mode: str = "all_history",  # 'all_history' or 'bounded'
        bounded_window: Optional[Tuple[int, int]] = None,
        history_weight: float = 1.0,
        cn_weight: float = 0.3
    ):
        self.mode = mode
        self.bounded_window = bounded_window
        self.history_weight = history_weight
        self.cn_weight = cn_weight

    def predict_pairs(
        self,
        G_t: np.ndarray,
        historical_snapshots: List[np.ndarray],
        pairs: np.ndarray,
        current_time: int,
        accessed_times: List[int]
    ) -> np.ndarray:
        """
        G_t: Current graph at timestep t.
        historical_snapshots: Historical graph snapshots.
        pairs: Candidate node pairs (u, v) to evaluate.
        current_time: Active evaluation timestep t.
        accessed_times: Timestamps accessed to form historical_snapshots.
        """
        verify_no_future_leakage(current_time, accessed_times)

        if len(pairs) == 0:
            return np.empty((0,), dtype=np.float64)

        u = pairs[:, 0]
        v = pairs[:, 1]
        current_edges = G_t[u, v].astype(np.float64)

        if len(historical_snapshots) == 0:
            cn_scores = np.sum(G_t[u] * G_t[v], axis=1).astype(np.float64)
            max_cn = np.max(cn_scores) if len(cn_scores) > 0 and np.max(cn_scores) > 0 else 1.0
            return current_edges + self.cn_weight * (cn_scores / max_cn)

        # Build exact historical edge occurrence mask (union across valid historical snapshots)
        if self.mode == "bounded" and self.bounded_window is not None:
            w_start, w_end = self.bounded_window
            valid_snaps = [snap for snap, t_h in zip(historical_snapshots, accessed_times) if w_start <= t_h <= w_end and t_h < current_time]
        else:
            valid_snaps = [snap for snap, t_h in zip(historical_snapshots, accessed_times) if t_h < current_time]

        if len(valid_snaps) == 0:
            hist_edges = np.zeros(len(pairs), dtype=np.float64)
            blended = G_t.astype(np.float64)
        else:
            H_union = np.maximum.reduce(valid_snaps).astype(np.float64)
            hist_edges = H_union[u, v]
            blended = G_t.astype(np.float64) + self.history_weight * H_union

        cn_scores = np.sum(blended[u] * blended[v], axis=1).astype(np.float64)
        max_cn = np.max(cn_scores) if len(cn_scores) > 0 and np.max(cn_scores) > 0 else 1.0

        scores = current_edges + self.history_weight * hist_edges + self.cn_weight * (cn_scores / max_cn)
        return scores
