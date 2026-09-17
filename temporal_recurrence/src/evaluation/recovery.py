"""Recovery latency evaluation following regime transition B -> A."""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from ..generator.dsbm import DynamicGraphSequence
from .prediction import sample_evaluation_edges, compute_prediction_metrics, verify_no_future_leakage
def evaluate_recovery_trajectory(
    seq: DynamicGraphSequence,
    stationary_reference_ap: float,
    recovery_start_time: int,
    threshold_ratio: float = 0.90,
    sustained_steps: int = 5,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate AP over tau = time since B -> A transition for:
    - current-only
    - recent-history (h=5)
    - historical oracle
    """
    from ..baselines.current_only import CurrentOnlyPredictor
    from ..baselines.recent_history import RecentHistoryPredictor
    from ..baselines.historical_oracle import HistoricalOraclePredictor

    current_pred = CurrentOnlyPredictor()
    recent_pred = RecentHistoryPredictor(window_size=5)
    oracle_pred = HistoricalOraclePredictor(variant="historical_summary")

    prev_A_episodes = seq.scheduler.get_previous_episodes_for_regime("A", before_time=recovery_start_time)
    if prev_A_episodes:
        last_A = prev_A_episodes[-1]
        hist_A_times = list(range(int(last_A["start"]), int(last_A["end"]) + 1))
        hist_A_snaps = [seq.get_snapshot(t) for t in hist_A_times]
    else:
        hist_A_times = []
        hist_A_snaps = []

    rec_record = seq.scheduler.get_record(recovery_start_time)
    end_time = rec_record.regime_end
    # We evaluate up to end_time - 1 (since target is t+1)
    max_eval_time = min(end_time, seq.total_timesteps - 2)

    target_ap_threshold = threshold_ratio * stationary_reference_ap

    records = []
    current_ap_list = []
    recent_ap_list = []
    oracle_ap_list = []

    for t in range(recovery_start_time, max_eval_time + 1):
        tau = t - recovery_start_time
        G_t = seq.get_snapshot(t)
        G_next = seq.get_snapshot(t + 1)

        pairs, labels = sample_evaluation_edges(G_next, seed=seed + tau)
        if len(labels) == 0:
            continue

        # Current-only
        c_scores = current_pred.predict_pairs(G_t, pairs)
        c_ap = compute_prediction_metrics(c_scores, labels)["ap"]

        # Recent-history (h=5)
        h_times = list(range(max(0, t - 4), t + 1))
        verify_no_future_leakage(t, h_times)
        h_snaps = [seq.get_snapshot(ti) for ti in h_times]
        r_scores = recent_pred.predict_pairs(h_snaps, pairs)
        r_ap = compute_prediction_metrics(r_scores, labels)["ap"]

        # Oracle
        o_scores = oracle_pred.predict_pairs(G_t, hist_A_snaps, pairs, current_time=t, accessed_times=hist_A_times)
        o_ap = compute_prediction_metrics(o_scores, labels)["ap"]

        records.append({
            "time": t,
            "tau": tau,
            "current_only_ap": c_ap,
            "recent_history_ap": r_ap,
            "historical_oracle_ap": o_ap,
            "target_threshold": target_ap_threshold
        })

        current_ap_list.append(c_ap)
        recent_ap_list.append(r_ap)
        oracle_ap_list.append(o_ap)

    df = pd.DataFrame(records)

    # Compute T_recover for each predictor
    def find_t_recover(ap_series: List[float]) -> int:
        for idx in range(len(ap_series) - sustained_steps + 1):
            window = ap_series[idx : idx + sustained_steps]
            if all(val >= target_ap_threshold for val in window):
                return idx
        return -1  # Did not recover within observation window

    t_recover_current = find_t_recover(current_ap_list)
    t_recover_recent = find_t_recover(recent_ap_list)
    t_recover_oracle = find_t_recover(oracle_ap_list)

    summary = {
        "stationary_reference_ap": stationary_reference_ap,
        "target_ap_threshold": target_ap_threshold,
        "t_recover_current": t_recover_current,
        "t_recover_recent": t_recover_recent,
        "t_recover_oracle": t_recover_oracle,
        "oracle_recovery_advantage": (t_recover_current - t_recover_oracle) if (t_recover_current >= 0 and t_recover_oracle >= 0) else None
    }

    return df, summary
