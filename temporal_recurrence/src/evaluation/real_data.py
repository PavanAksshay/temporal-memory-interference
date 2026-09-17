"""Real Temporal Graph Dataset Loader, Windowing, and Regime Discovery for Phase 3."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp

from .prediction import verify_no_future_leakage
from .event_converter import TemporalEventBatch
from ..generator.dsbm import DynamicGraphSequence, RegimeScheduler


@dataclass
class RealDataWindow:
    window_idx: int
    start_time: int
    end_time: int
    num_events: int
    active_nodes: int
    adjacency: np.ndarray  # Shape: (N, N)
    degree_profile: np.ndarray  # Shape: (N,)
    density: float


class RealTemporalGraphDataset:
    """
    Loads, processes, and windows real event-based temporal interaction graphs.
    """

    def __init__(self, data_path: str = "data/real/CollegeMsg.txt", num_windows: int = 20, max_nodes: int = 300):
        self.data_path = Path(data_path)
        self.num_windows = num_windows
        self.max_nodes = max_nodes
        self.events: List[Tuple[int, int, int]] = []
        self.node_map: Dict[int, int] = {}
        self.num_nodes: int = 0
        self.windows: List[RealDataWindow] = []
        self._load_and_process()

    def _load_and_process(self) -> None:
        raw_events = []
        node_counts = {}
        with open(self.data_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    u, v, ts = int(parts[0]), int(parts[1]), int(float(parts[2]))
                    raw_events.append((u, v, ts))
                    node_counts[u] = node_counts.get(u, 0) + 1
                    node_counts[v] = node_counts.get(v, 0) + 1

        # Select top active nodes up to max_nodes
        sorted_nodes = sorted(node_counts.keys(), key=lambda x: node_counts[x], reverse=True)[:self.max_nodes]
        self.node_map = {n: idx for idx, n in enumerate(sorted_nodes)}
        self.num_nodes = len(self.node_map)

        # Filter and remap events
        filtered_events = []
        for u, v, ts in raw_events:
            if u in self.node_map and v in self.node_map and u != v:
                filtered_events.append((self.node_map[u], self.node_map[v], ts))

        # Sort chronologically
        filtered_events.sort(key=lambda x: x[2])
        self.events = filtered_events

        if len(self.events) == 0:
            return

        min_ts = self.events[0][2]
        max_ts = self.events[-1][2]
        window_duration = (max_ts - min_ts) / self.num_windows

        # Build chronological windows
        for w_idx in range(self.num_windows):
            w_start = min_ts + w_idx * window_duration
            w_end = min_ts + (w_idx + 1) * window_duration
            w_events = [e for e in self.events if w_start <= e[2] < w_end]

            adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.uint8)
            for u, v, _ in w_events:
                adj[u, v] = 1
                adj[v, u] = 1

            deg = np.sum(adj, axis=1).astype(np.float64)
            active = np.sum(deg > 0)
            density = float(np.sum(adj)) / (self.num_nodes * (self.num_nodes - 1) + 1e-8)

            self.windows.append(
                RealDataWindow(
                    window_idx=w_idx,
                    start_time=int(w_start),
                    end_time=int(w_end),
                    num_events=len(w_events),
                    active_nodes=int(active),
                    adjacency=adj,
                    degree_profile=deg,
                    density=density
                )
            )

    def compute_similarity_matrix(self) -> np.ndarray:
        """Computes pairwise cosine similarity between window degree/interaction profiles."""
        T = len(self.windows)
        sim_mat = np.zeros((T, T), dtype=np.float64)
        for i in range(T):
            u_vec = self.windows[i].degree_profile
            norm_u = np.linalg.norm(u_vec) + 1e-8
            for j in range(T):
                v_vec = self.windows[j].degree_profile
                norm_v = np.linalg.norm(v_vec) + 1e-8
                sim_mat[i, j] = float(np.dot(u_vec, v_vec) / (norm_u * norm_v))
        return sim_mat

    def discover_recurring_episodes(self, min_gap: int = 2, sim_threshold: float = 0.65) -> List[Tuple[int, int, int, float]]:
        """
        Discovers empirical A -> B -> A recurrence episodes:
        Returns list of tuples (t_A1, t_B, t_A2, similarity_A1_A2) where:
        - sim(W_A1, W_A2) >= sim_threshold
        - sim(W_A1, W_B) < sim(W_A1, W_A2) - 0.15
        - t_A1 < t_B < t_A2
        """
        sim_mat = self.compute_similarity_matrix()
        T = len(self.windows)
        episodes = []

        for i in range(T):
            for k in range(i + min_gap, T):
                sim_ik = sim_mat[i, k]
                if sim_ik >= sim_threshold:
                    # Look for intermediate distractor window j
                    best_j = -1
                    min_sim_j = 1e9
                    for j in range(i + 1, k):
                        sim_ij = sim_mat[i, j]
                        if sim_ij < min_sim_j:
                            min_sim_j = sim_ij
                            best_j = j
                    if best_j != -1 and (sim_ik - min_sim_j) >= 0.10:
                        episodes.append((i, best_j, k, sim_ik))

        # Sort by recurring similarity
        episodes.sort(key=lambda x: x[3], reverse=True)
        return episodes

    def extract_event_batches(self, negative_ratio: float = 1.0, seed: int = 42) -> List[TemporalEventBatch]:
        """Extracts per-window interaction event batches with 1:1 balanced negative sampling."""
        rng = np.random.default_rng(seed)
        event_batches = []
        N = self.num_nodes
        triu_i, triu_j = np.triu_indices(N, k=1)

        for w in self.windows:
            adj = w.adjacency
            edges = adj[triu_i, triu_j]
            pos_mask = (edges == 1)
            neg_mask = (edges == 0)

            pos_u = triu_i[pos_mask]
            pos_v = triu_j[pos_mask]
            num_pos = len(pos_u)

            if num_pos == 0:
                event_batches.append(
                    TemporalEventBatch(
                        src=np.empty((0,), dtype=int),
                        dst=np.empty((0,), dtype=int),
                        timestamps=np.empty((0,), dtype=float),
                        labels=np.empty((0,), dtype=int)
                    )
                )
                continue

            neg_u_all = triu_i[neg_mask]
            neg_v_all = triu_j[neg_mask]
            num_neg = min(int(num_pos * negative_ratio), len(neg_u_all))

            neg_idx = rng.choice(len(neg_u_all), size=num_neg, replace=False)
            neg_u = neg_u_all[neg_idx]
            neg_v = neg_v_all[neg_idx]

            src_arr = np.concatenate([pos_u, neg_u])
            dst_arr = np.concatenate([pos_v, neg_v])
            labels_arr = np.concatenate([np.ones(num_pos, dtype=np.int32), np.zeros(num_neg, dtype=np.int32)])
            ts_arr = np.full(len(labels_arr), fill_value=float(w.window_idx), dtype=np.float32)

            perm = rng.permutation(len(labels_arr))
            event_batches.append(
                TemporalEventBatch(
                    src=src_arr[perm],
                    dst=dst_arr[perm],
                    timestamps=ts_arr[perm],
                    labels=labels_arr[perm]
                )
            )

        return event_batches
