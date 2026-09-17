"""Regime definitions, structural affinity matrices, independent partitions, and Markov transition parameters."""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class RegimeConfig:
    name: str
    target_density: float
    persistence: float  # lambda in [0, 1)
    within_comm_multiplier: float  # structural affinity multiplier (w_in / w_out)
    partition_seed: Optional[int] = None  # Seed for independent community partition


class Regime:
    """Represents a dynamic graph regime with transition parameters and stationary affinity."""

    def __init__(
        self,
        config: RegimeConfig,
        num_nodes: int,
        num_communities: int,
        community_assignments: Optional[np.ndarray] = None,
        mode: str = "temporal_plus_structure"
    ):
        self.config = config
        self.name = config.name
        self.target_density = float(config.target_density)
        self.persistence = float(config.persistence)
        self.within_comm_multiplier = float(config.within_comm_multiplier)
        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.mode = mode

        # Community partition handling
        if community_assignments is not None:
            self.community_assignments = community_assignments
        elif config.partition_seed is not None:
            rng = np.random.default_rng(config.partition_seed)
            nodes_per_comm = num_nodes // num_communities
            c = []
            for k in range(num_communities):
                count = nodes_per_comm if k < num_communities - 1 else (num_nodes - len(c))
                c.extend([k] * count)
            c = np.array(c, dtype=np.int32)
            rng.shuffle(c)
            self.community_assignments = c
        else:
            nodes_per_comm = num_nodes // num_communities
            c = []
            for k in range(num_communities):
                count = nodes_per_comm if k < num_communities - 1 else (num_nodes - len(c))
                c.extend([k] * count)
            self.community_assignments = np.array(c, dtype=np.int32)

        self.affinity_matrix = self._compute_affinity_matrix()
        self.a_matrix, self.b_matrix = self._compute_transition_matrices()

    def _compute_affinity_matrix(self) -> np.ndarray:
        """Compute the stationary edge probability matrix W in [0, 1]^(N x N)."""
        N = self.num_nodes
        W = np.zeros((N, N), dtype=np.float64)

        if self.mode == "temporal_only" or self.within_comm_multiplier == 1.0:
            np.fill_diagonal(W, 0.0)
            mask = ~np.eye(N, dtype=bool)
            W[mask] = self.target_density
            return W

        # Structural mode: block-dependent affinities
        c = self.community_assignments
        same_comm_mask = (c[:, None] == c[None, :]) & (~np.eye(N, dtype=bool))
        diff_comm_mask = (c[:, None] != c[None, :])

        E_in = np.sum(same_comm_mask) / 2.0
        E_out = np.sum(diff_comm_mask) / 2.0
        total_possible_edges = N * (N - 1) / 2.0

        M = self.within_comm_multiplier
        # E_in * (M * w_out) + E_out * w_out = total_possible_edges * target_density
        w_out = (total_possible_edges * self.target_density) / (M * E_in + E_out)
        w_in = M * w_out

        # Clip for numerical stability
        w_out = float(np.clip(w_out, 0.001, 0.999))
        w_in = float(np.clip(w_in, 0.001, 0.999))

        W[same_comm_mask] = w_in
        W[diff_comm_mask] = w_out
        np.fill_diagonal(W, 0.0)

        # Scale slightly to match target density exactly
        actual_mean_density = np.sum(W) / (2.0 * total_possible_edges)
        if actual_mean_density > 0:
            scale = self.target_density / actual_mean_density
            W = np.clip(W * scale, 0.0, 0.999)
            np.fill_diagonal(W, 0.0)

        return W

    def _compute_transition_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute transition probability matrices:
        a_ij = P(X_{t+1}=1 | X_t=0) = W_ij * (1 - lambda)
        b_ij = P(X_{t+1}=0 | X_t=1) = (1 - W_ij) * (1 - lambda)
        Stationary probability pi_ij = a_ij / (a_ij + b_ij) = W_ij.
        """
        lam = self.persistence
        a = self.affinity_matrix * (1.0 - lam)
        b = (1.0 - self.affinity_matrix) * (1.0 - lam)
        np.fill_diagonal(a, 0.0)
        np.fill_diagonal(b, 1.0)
        return a, b


def compute_regime_affinity_similarity(regime1: Regime, regime2: Regime) -> float:
    """Compute cosine similarity between upper-triangular affinity matrices of two regimes."""
    N = regime1.num_nodes
    triu_idx = np.triu_indices(N, k=1)
    w1 = regime1.affinity_matrix[triu_idx]
    w2 = regime2.affinity_matrix[triu_idx]

    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(w1, w2) / (norm1 * norm2))
