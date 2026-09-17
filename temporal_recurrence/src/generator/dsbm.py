"""Dynamic Stochastic Block Model (DSBM) generator with regime scheduling."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .regimes import Regime, RegimeConfig


@dataclass
class RegimeTransitionRecord:
    time: int
    regime: str
    regime_start: int
    regime_end: int
    time_since_transition: int
    episode_index: int


class RegimeScheduler:
    """Manages multi-regime timeline scheduling and transition tracking."""

    def __init__(self, sequence_spec: List[Tuple[str, int]]):
        """
        sequence_spec: List of tuples like [("A", 100), ("B", 200), ("A", 100)]
        """
        self.sequence_spec = sequence_spec
        self.timeline: List[RegimeTransitionRecord] = []
        self.episodes: List[Dict[str, int | str]] = []
        self._build_schedule()

    def _build_schedule(self) -> None:
        current_time = 0
        for ep_idx, (regime_name, duration) in enumerate(self.sequence_spec):
            start = current_time
            end = current_time + duration - 1
            self.episodes.append({
                "episode_index": ep_idx,
                "regime": regime_name,
                "start": start,
                "end": end,
                "duration": duration
            })
            for t in range(start, end + 1):
                self.timeline.append(
                    RegimeTransitionRecord(
                        time=t,
                        regime=regime_name,
                        regime_start=start,
                        regime_end=end,
                        time_since_transition=t - start,
                        episode_index=ep_idx
                    )
                )
            current_time += duration

    @property
    def total_timesteps(self) -> int:
        return len(self.timeline)

    def get_record(self, t: int) -> RegimeTransitionRecord:
        if 0 <= t < len(self.timeline):
            return self.timeline[t]
        raise IndexError(f"Timestep {t} out of range [0, {len(self.timeline)-1}]")

    def get_previous_episodes_for_regime(self, regime_name: str, before_time: int) -> List[Dict[str, int | str]]:
        """Return completed episodes for regime_name strictly ending before before_time."""
        return [
            ep for ep in self.episodes
            if ep["regime"] == regime_name and int(ep["end"]) < before_time
        ]


@dataclass
class DynamicGraphSequence:
    """Container holding temporal graph snapshots and regime metadata."""
    snapshots: np.ndarray  # Shape: (T, N, N), boolean/uint8
    num_nodes: int
    num_communities: int
    community_assignments: np.ndarray
    scheduler: RegimeScheduler
    regimes: Dict[str, Regime]

    @property
    def total_timesteps(self) -> int:
        return self.snapshots.shape[0]

    def get_snapshot(self, t: int) -> np.ndarray:
        return self.snapshots[t]


class DynamicSBMGenerator:
    """Generates continuous snapshot sequences across regime schedules."""

    def __init__(
        self,
        num_nodes: int = 300,
        num_communities: int = 3,
        mode: str = "temporal_plus_structure",
        regime_configs: Optional[Dict[str, RegimeConfig]] = None
    ):
        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.mode = mode

        # Default community assignments: equal size partitions
        nodes_per_comm = num_nodes // num_communities
        c = []
        for k in range(num_communities):
            count = nodes_per_comm if k < num_communities - 1 else (num_nodes - len(c))
            c.extend([k] * count)
        self.community_assignments = np.array(c, dtype=np.int32)

        # Build regimes
        if regime_configs is None:
            regime_configs = {
                "A": RegimeConfig("A", target_density=0.10, persistence=0.85, within_comm_multiplier=3.5),
                "B": RegimeConfig("B", target_density=0.10, persistence=0.20, within_comm_multiplier=0.8),
                "C": RegimeConfig("C", target_density=0.10, persistence=0.50, within_comm_multiplier=1.5),
            }

        self.regimes: Dict[str, Regime] = {
            name: Regime(
                cfg,
                num_nodes=self.num_nodes,
                num_communities=self.num_communities,
                community_assignments=None if cfg.partition_seed is not None else self.community_assignments,
                mode=self.mode
            )
            for name, cfg in regime_configs.items()
        }

    def generate(
        self,
        sequence_spec: List[Tuple[str, int]],
        seed: int = 42
    ) -> DynamicGraphSequence:
        """
        Generate temporal graph snapshots across the specified regime sequence.
        """
        rng = np.random.default_rng(seed)
        scheduler = RegimeScheduler(sequence_spec)
        T = scheduler.total_timesteps
        N = self.num_nodes

        snapshots = np.zeros((T, N, N), dtype=np.uint8)

        # Step 0: sample from stationary distribution of initial regime
        initial_regime_name = scheduler.get_record(0).regime
        initial_regime = self.regimes[initial_regime_name]
        W0 = initial_regime.affinity_matrix

        # Generate upper triangle Bernoulli draws
        triu_indices = np.triu_indices(N, k=1)
        p0_upper = W0[triu_indices]
        draw0 = (rng.random(len(p0_upper)) < p0_upper).astype(np.uint8)

        G0 = np.zeros((N, N), dtype=np.uint8)
        G0[triu_indices] = draw0
        G0 = G0 + G0.T  # Symmetric undirected
        snapshots[0] = G0

        # Subsequent timesteps: Markov transitions conditional on G_{t-1} and active regime
        for t in range(1, T):
            active_regime_name = scheduler.get_record(t).regime
            active_regime = self.regimes[active_regime_name]
            a_mat = active_regime.a_matrix
            b_mat = active_regime.b_matrix

            prev_state = snapshots[t - 1][triu_indices]
            a_vals = a_mat[triu_indices]
            b_vals = b_mat[triu_indices]

            # P(X_{t}=1 | X_{t-1}) = X_{t-1}*(1-b) + (1-X_{t-1})*a
            p_transition = prev_state * (1.0 - b_vals) + (1.0 - prev_state) * a_vals

            rand_draws = rng.random(len(p_transition))
            next_state = (rand_draws < p_transition).astype(np.uint8)

            Gt = np.zeros((N, N), dtype=np.uint8)
            Gt[triu_indices] = next_state
            Gt = Gt + Gt.T
            snapshots[t] = Gt

        return DynamicGraphSequence(
            snapshots=snapshots,
            num_nodes=N,
            num_communities=self.num_communities,
            community_assignments=self.community_assignments,
            scheduler=scheduler,
            regimes=self.regimes
        )
