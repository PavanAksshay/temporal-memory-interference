"""Historical relevance evaluation: R(k) = AP(G_t + G_{t-k}) - AP(G_t)."""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from ..generator.dsbm import DynamicGraphSequence
from .prediction import sample_evaluation_edges, compute_prediction_metrics, verify_no_future_leakage
from ..baselines.current_only import CurrentOnlyPredictor


def evaluate_lag_relevance(
    seq: DynamicGraphSequence,
    target_time: int,
    candidate_lags: List[int],
    seed: int = 42
) -> pd.DataFrame:
    """
    Evaluate predictive relevance R(k) = AP(current + lag k) - AP(current) at a specific time t.
    """
    if target_time >= seq.total_timesteps - 1:
        raise ValueError(f"target_time {target_time} must be < total_timesteps - 1 ({seq.total_timesteps - 1})")

    G_t = seq.get_snapshot(target_time)
    G_next = seq.get_snapshot(target_time + 1)

    pairs, labels = sample_evaluation_edges(G_next, seed=seed)
    if len(labels) == 0:
        return pd.DataFrame()

    # Current only score
    current_predictor = CurrentOnlyPredictor()
    current_scores = current_predictor.predict_pairs(G_t, pairs)
    current_metrics = compute_prediction_metrics(current_scores, labels)
    current_ap = current_metrics["ap"]
    current_auc = current_metrics["auc"]

    results = []
    u, v = pairs[:, 0], pairs[:, 1]

    for lag in candidate_lags:
        lag_time = target_time - lag
        if lag_time < 0:
            continue

        verify_no_future_leakage(target_time, [lag_time])
        G_lag = seq.get_snapshot(lag_time)
        lag_regime = seq.scheduler.get_record(lag_time).regime

        # Combine current and historical lag
        lag_edges = G_lag[u, v].astype(np.float64)
        combined_scores = current_scores + 0.8 * lag_edges

        metrics = compute_prediction_metrics(combined_scores, labels)
        r_k_ap = metrics["ap"] - current_ap
        r_k_auc = metrics["auc"] - current_auc

        results.append({
            "target_time": target_time,
            "lag_k": lag,
            "lag_time": lag_time,
            "lag_regime": lag_regime,
            "current_ap": current_ap,
            "lag_ap": metrics["ap"],
            "relevance_r_k_ap": r_k_ap,
            "current_auc": current_auc,
            "lag_auc": metrics["auc"],
            "relevance_r_k_auc": r_k_auc
        })

    return pd.DataFrame(results)


def evaluate_recurrence_conflict(
    seq: DynamicGraphSequence,
    recurrence_start_time: int,
    window_length: int = 30,
    seed: int = 42
) -> Dict[str, float]:
    """
    Measure Delta_recent (recent B) and Delta_old (previous A) immediately after B -> A transition.
    """
    from ..baselines.historical_oracle import HistoricalOraclePredictor
    current_pred = CurrentOnlyPredictor()
    oracle_pred = HistoricalOraclePredictor(variant="historical_summary")

    delta_recent_list = []
    delta_old_list = []

    # Identify previous A episode
    prev_A_episodes = seq.scheduler.get_previous_episodes_for_regime("A", before_time=recurrence_start_time)
    if not prev_A_episodes:
        return {"delta_old": 0.0, "delta_recent": 0.0, "delta_net": 0.0}

    last_A_ep = prev_A_episodes[-1]
    hist_A_times = list(range(int(last_A_ep["start"]), int(last_A_ep["end"]) + 1))
    hist_A_snaps = [seq.get_snapshot(t) for t in hist_A_times]

    eval_end = min(recurrence_start_time + window_length, seq.total_timesteps - 1)

    for step_idx, t in enumerate(range(recurrence_start_time, eval_end)):
        G_t = seq.get_snapshot(t)
        G_next = seq.get_snapshot(t + 1)
        pairs, labels = sample_evaluation_edges(G_next, seed=seed + step_idx)
        if len(labels) == 0:
            continue

        u, v = pairs[:, 0], pairs[:, 1]
        c_scores = current_pred.predict_pairs(G_t, pairs)
        c_ap = compute_prediction_metrics(c_scores, labels)["ap"]

        # Oracle (Historical A)
        oracle_scores = oracle_pred.predict_pairs(G_t, hist_A_snaps, pairs, current_time=t, accessed_times=hist_A_times)
        o_ap = compute_prediction_metrics(oracle_scores, labels)["ap"]
        delta_old_list.append(o_ap - c_ap)

        # Recent B (lag from recent B regime, e.g. 5 steps ago if available and strictly in B)
        recent_b_time = t - 5
        if recent_b_time >= 0 and seq.scheduler.get_record(recent_b_time).regime == "B":
            verify_no_future_leakage(t, [recent_b_time])
            G_b = seq.get_snapshot(recent_b_time)
            recent_b_scores = c_scores + 0.8 * G_b[u, v].astype(np.float64)
            b_ap = compute_prediction_metrics(recent_b_scores, labels)["ap"]
            delta_recent_list.append(b_ap - c_ap)
        else:
            # If t - 5 is not in B, use the last snapshot of B (t_switch - 1)
            b_last_time = recurrence_start_time - 1
            if b_last_time >= 0:
                verify_no_future_leakage(t, [b_last_time])
                G_b = seq.get_snapshot(b_last_time)
                recent_b_scores = c_scores + 0.8 * G_b[u, v].astype(np.float64)
                b_ap = compute_prediction_metrics(recent_b_scores, labels)["ap"]
                delta_recent_list.append(b_ap - c_ap)

    mean_delta_old = float(np.mean(delta_old_list)) if delta_old_list else 0.0
    mean_delta_recent = float(np.mean(delta_recent_list)) if delta_recent_list else 0.0

    return {
        "delta_old": mean_delta_old,
        "delta_recent": mean_delta_recent,
        "delta_net": mean_delta_old - mean_delta_recent
    }
