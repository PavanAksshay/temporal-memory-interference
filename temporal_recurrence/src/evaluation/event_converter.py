"""Utilities for converting graph snapshots into chronological event streams."""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch

from ..generator.dsbm import DynamicGraphSequence


@dataclass
class TemporalEventBatch:
    src: np.ndarray
    dst: np.ndarray
    timestamps: np.ndarray
    labels: np.ndarray  # 1 for positive, 0 for negative


def extract_events_from_sequence(
    seq: DynamicGraphSequence,
    negative_ratio: float = 1.0,
    seed: int = 42
) -> List[TemporalEventBatch]:
    """
    Extract per-timestep interaction events and matched negative edges from dynamic graph sequence.
    Each timestep t provides positive edges existing at t and negative edges absent at t.
    """
    rng = np.random.default_rng(seed)
    T = seq.total_timesteps
    N = seq.num_nodes
    triu_i, triu_j = np.triu_indices(N, k=1)

    event_batches = []

    for t in range(T):
        G_t = seq.get_snapshot(t)
        edges = G_t[triu_i, triu_j]

        pos_mask = (edges == 1)
        neg_mask = (edges == 0)

        pos_u = triu_i[pos_mask]
        pos_v = triu_j[pos_mask]
        num_pos = len(pos_u)

        if num_pos == 0:
            event_batches.append(TemporalEventBatch(
                src=np.empty((0,), dtype=int),
                dst=np.empty((0,), dtype=int),
                timestamps=np.empty((0,), dtype=float),
                labels=np.empty((0,), dtype=int)
            ))
            continue

        neg_u_all = triu_i[neg_mask]
        neg_v_all = triu_j[neg_mask]
        num_neg = min(int(num_pos * negative_ratio), len(neg_u_all))

        neg_idx = rng.choice(len(neg_u_all), size=num_neg, replace=False)
        neg_u = neg_u_all[neg_idx]
        neg_v = neg_v_all[neg_idx]

        src = np.concatenate([pos_u, neg_u])
        dst = np.concatenate([pos_v, neg_v])
        labels = np.concatenate([np.ones(num_pos, dtype=np.int32), np.zeros(num_neg, dtype=np.int32)])
        ts = np.full(len(src), float(t), dtype=np.float32)

        # Shuffle pairs within timestep
        perm = rng.permutation(len(src))
        event_batches.append(TemporalEventBatch(
            src=src[perm],
            dst=dst[perm],
            timestamps=ts[perm],
            labels=labels[perm]
        ))

    return event_batches
