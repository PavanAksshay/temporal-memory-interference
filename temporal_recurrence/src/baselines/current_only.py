"""Current-only baseline predictor using only information from G_t."""
import numpy as np


class CurrentOnlyPredictor:
    """Predicts future edge states Y_{e, t+1} using strictly the current snapshot G_t."""

    def __init__(self, cn_weight: float = 0.5):
        self.cn_weight = cn_weight

    def predict_pairs(self, G_t: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        """
        Compute edge scores for given node pairs using only G_t.
        pairs: shape (M, 2)
        """
        if len(pairs) == 0:
            return np.empty((0,), dtype=np.float64)

        u = pairs[:, 0]
        v = pairs[:, 1]

        # Direct current edge state
        current_edges = G_t[u, v].astype(np.float64)

        # Common neighbors in G_t
        # CN(u, v) = (G_t @ G_t)[u, v]
        # For efficiency on pairs:
        # dot product of rows u and v
        cn_scores = np.sum(G_t[u] * G_t[v], axis=1).astype(np.float64)
        max_cn = np.max(cn_scores) if len(cn_scores) > 0 and np.max(cn_scores) > 0 else 1.0

        scores = current_edges + self.cn_weight * (cn_scores / max_cn)
        return scores
