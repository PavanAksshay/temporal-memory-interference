"""Simple Historical Retrieval Predictor and State Cache for Phase 2."""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..evaluation.prediction import verify_no_future_leakage


class HistoricalStateCache:
    """
    Maintains historical graph representations/snapshots with strict timestamp tracking
    to guarantee zero future leakage.
    """

    def __init__(self):
        self.snapshots: Dict[int, np.ndarray] = {}
        self.embeddings: Dict[int, np.ndarray] = {}
        self.regime_tags: Dict[int, str] = {}

    def store_state(self, t: int, graph_snapshot: np.ndarray, regime_tag: str = "A", embedding: Optional[np.ndarray] = None) -> None:
        self.snapshots[t] = graph_snapshot.copy()
        self.regime_tags[t] = regime_tag
        if embedding is not None:
            self.embeddings[t] = embedding.copy()
        else:
            # Default summary embedding: normalized degree profile / adjacency row summary
            deg = np.sum(graph_snapshot, axis=1) / (graph_snapshot.shape[0] - 1 + 1e-8)
            self.embeddings[t] = deg

    def get_valid_history(self, current_time: int) -> List[int]:
        """Return all historical timestamps strictly prior to current_time."""
        return [t for t in sorted(self.snapshots.keys()) if t < current_time]

    def get_regime_history(self, current_time: int, regime_tag: str) -> List[int]:
        """Return historical timestamps for a specific regime strictly prior to current_time."""
        return [t for t in sorted(self.snapshots.keys()) if t < current_time and self.regime_tags[t] == regime_tag]


class HistoricalRetrievalPredictor:
    """
    Evaluates link prediction under distinct historical retrieval modes:
    - R1: Oracle Retrieval (strictly retrieves from historical relevant regime A)
    - R2: Similarity Retrieval (retrieves highest-cosine-similarity historical state)
    - R3: Random Retrieval (randomly selects an old historical state)
    - Recent-B: Misleading Retrieval (strictly retrieves from distractor regime B)
    """

    def __init__(
        self,
        mode: str = "oracle",  # "oracle", "similarity", "random", "recent_b"
        history_weight: float = 1.0,
        cn_weight: float = 0.3
    ):
        self.mode = mode
        self.history_weight = history_weight
        self.cn_weight = cn_weight

    def retrieve_state(
        self,
        current_time: int,
        current_graph: np.ndarray,
        cache: HistoricalStateCache,
        seed: int = 42
    ) -> Tuple[np.ndarray, int]:
        """
        Retrieves historical state and verified timestamp based on mode.
        Returns: (retrieved_snapshot, retrieved_timestamp)
        """
        valid_times = cache.get_valid_history(current_time)
        if len(valid_times) == 0:
            return np.zeros_like(current_graph), -1

        if self.mode == "oracle":
            # R1: Retrieve from historical relevant regime A
            a_times = cache.get_regime_history(current_time, "A")
            if len(a_times) == 0:
                selected_t = valid_times[0]
            else:
                selected_t = a_times[-1]  # Most recent relevant A snapshot
            verify_no_future_leakage(current_time, [selected_t])
            return cache.snapshots[selected_t], selected_t

        elif self.mode == "similarity":
            # R2: Retrieve state with highest cosine similarity to current graph summary
            current_summary = np.sum(current_graph, axis=1) / (current_graph.shape[0] - 1 + 1e-8)
            norm_curr = np.linalg.norm(current_summary) + 1e-8
            best_sim = -1e9
            best_t = valid_times[0]
            for t in valid_times:
                h_emb = cache.embeddings[t]
                norm_h = np.linalg.norm(h_emb) + 1e-8
                sim = np.dot(current_summary, h_emb) / (norm_curr * norm_h)
                if sim > best_sim:
                    best_sim = sim
                    best_t = t
            verify_no_future_leakage(current_time, [best_t])
            return cache.snapshots[best_t], best_t

        elif self.mode == "random":
            # R3: Random historical state
            rng = np.random.default_rng(seed + current_time)
            selected_t = int(rng.choice(valid_times))
            verify_no_future_leakage(current_time, [selected_t])
            return cache.snapshots[selected_t], selected_t

        elif self.mode == "recent_b":
            # Distractor control: strictly from regime B
            b_times = cache.get_regime_history(current_time, "B")
            if len(b_times) == 0:
                selected_t = valid_times[-1]
            else:
                selected_t = b_times[-1]
            verify_no_future_leakage(current_time, [selected_t])
            return cache.snapshots[selected_t], selected_t

        else:
            raise ValueError(f"Unknown retrieval mode: {self.mode}")

    def predict_pairs(
        self,
        G_t: np.ndarray,
        retrieved_H: np.ndarray,
        pairs: np.ndarray
    ) -> np.ndarray:
        """Score edge pairs combining current graph and retrieved history."""
        if len(pairs) == 0:
            return np.empty((0,), dtype=np.float64)

        u = pairs[:, 0]
        v = pairs[:, 1]

        current_edges = G_t[u, v].astype(np.float64)
        hist_edges = retrieved_H[u, v].astype(np.float64)

        blended = G_t.astype(np.float64) + self.history_weight * retrieved_H.astype(np.float64)
        cn_scores = np.sum(blended[u] * blended[v], axis=1).astype(np.float64)
        max_cn = np.max(cn_scores) if len(cn_scores) > 0 and np.max(cn_scores) > 0 else 1.0

        scores = current_edges + self.history_weight * hist_edges + self.cn_weight * (cn_scores / max_cn)
        return scores
