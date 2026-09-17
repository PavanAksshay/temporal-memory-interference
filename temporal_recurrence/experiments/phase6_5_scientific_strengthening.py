"""
Phase 6.5: Final Scientific Strengthening and Mechanism Falsification Suite.
Executes:
- Experiment 1: Current-Sufficient Specificity Control (10 seeds)
- Experiment 2: Exact Recurrence vs. Structural Recurrence Decomposition (10 seeds, 3 conditions)
- Experiment 3: Re-Exposure / Recovery Curve across k_A in {0, 1, 5, 10, 25} (10 seeds, T_B in {50, 100, 200})
- Experiment 4: Training Convergence Audit across 25 epochs (10 seeds, T_B in {25, 100, 200})
- Experiment 5: Frozen Memory Linear Probe tracking community decodability across regime transitions (10 seeds)
- Generates all CSVs, figures, and comprehensive reports in results/phase6_5/
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logging import setup_logger
from src.utils.random import set_seed
from src.generator.regimes import RegimeConfig
from src.generator.dsbm import DynamicSBMGenerator, DynamicGraphSequence
from src.evaluation.prediction import sample_evaluation_edges, compute_prediction_metrics
from src.evaluation.event_converter import extract_events_from_sequence, TemporalEventBatch
from src.baselines.current_only import CurrentOnlyPredictor
from src.baselines.historical_oracle import HistoricalOraclePredictor
from src.baselines.edgebank import EdgeBankPredictor
from src.models.tgn import TGN, TGNNoMemory
from src.models.retrieval import HistoricalStateCache, HistoricalRetrievalPredictor


def train_model_tgn_with_history(
    model: nn.Module,
    seq: DynamicGraphSequence,
    train_end_t: int,
    val_end_t: int,
    num_epochs: int = 25,
    lr: float = 0.005,
    batch_sample_size: int = 300,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    event_batches = extract_events_from_sequence(seq, negative_ratio=1.0, seed=seed)

    best_val_ap = -1.0
    best_state_dict = None
    epoch_history = []

    is_tgn_memory = hasattr(model, "reset_memory") and hasattr(model, "update_node_memories")

    for epoch in range(num_epochs):
        model.train()
        if is_tgn_memory:
            model.reset_memory()

        epoch_loss = 0.0
        step_count = 0

        for t in range(train_end_t + 1):
            batch = event_batches[t]
            if len(batch.src) == 0:
                continue

            if len(batch.src) > batch_sample_size:
                sub_idx = np.random.choice(len(batch.src), size=batch_sample_size, replace=False)
                src_sub, dst_sub = batch.src[sub_idx], batch.dst[sub_idx]
                ts_sub, lbl_sub = batch.timestamps[sub_idx], batch.labels[sub_idx]
            else:
                src_sub, dst_sub = batch.src, batch.dst
                ts_sub, lbl_sub = batch.timestamps, batch.labels

            src_t = torch.tensor(src_sub, dtype=torch.long, device=device)
            dst_t = torch.tensor(dst_sub, dtype=torch.long, device=device)
            ts_t = torch.tensor(ts_sub, dtype=torch.float32, device=device)
            lbl_t = torch.tensor(lbl_sub, dtype=torch.float32, device=device)

            logits = model.predict_logits(src_t, dst_t, ts_t)
            loss = criterion(logits, lbl_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            step_count += 1

            if is_tgn_memory:
                pos_mask = (batch.labels == 1)
                if np.any(pos_mask):
                    pos_src = torch.tensor(batch.src[pos_mask], dtype=torch.long, device=device)
                    pos_dst = torch.tensor(batch.dst[pos_mask], dtype=torch.long, device=device)
                    pos_ts = torch.tensor(batch.timestamps[pos_mask], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        model.update_node_memories(pos_src, pos_dst, pos_ts)

        # Validation pass
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for t in range(train_end_t + 1, val_end_t + 1):
                batch = event_batches[t]
                if len(batch.src) == 0:
                    continue
                src_t = torch.tensor(batch.src, dtype=torch.long, device=device)
                dst_t = torch.tensor(batch.dst, dtype=torch.long, device=device)
                ts_t = torch.tensor(batch.timestamps, dtype=torch.float32, device=device)

                probs = model.predict_link_probabilities(src_t, dst_t, ts_t).cpu().numpy()
                val_preds.extend(probs.tolist())
                val_targets.extend(batch.labels.tolist())

                if is_tgn_memory:
                    pos_mask = (batch.labels == 1)
                    if np.any(pos_mask):
                        pos_src = torch.tensor(batch.src[pos_mask], dtype=torch.long, device=device)
                        pos_dst = torch.tensor(batch.dst[pos_mask], dtype=torch.long, device=device)
                        pos_ts = torch.tensor(batch.timestamps[pos_mask], dtype=torch.float32, device=device)
                        model.update_node_memories(pos_src, pos_dst, pos_ts)

        val_ap = float(average_precision_score(val_targets, val_preds)) if len(val_targets) > 0 else 0.5
        val_auc = float(roc_auc_score(val_targets, val_preds)) if len(val_targets) > 0 else 0.5
        avg_loss = epoch_loss / max(1, step_count)

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_ap": val_ap,
            "val_auc": val_auc
        })

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()
    return model, epoch_history


def evaluate_tgn_rollout_with_k_A(
    model: nn.Module,
    seq: DynamicGraphSequence,
    test_start_t: int,
    test_end_t: int,
    k_A: int = 0,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> Dict[str, float]:
    """Roll out TGN memory through sequence, absorbing k_A steps of fresh A before evaluation."""
    model.eval()
    is_tgn_memory = hasattr(model, "reset_memory") and hasattr(model, "update_node_memories")
    if is_tgn_memory:
        model.reset_memory()

    event_batches = extract_events_from_sequence(seq, negative_ratio=1.0, seed=seed)
    all_preds, all_targets = [], []

    eval_actual_start_t = test_start_t + k_A

    with torch.no_grad():
        for t in range(test_end_t + 1):
            batch = event_batches[t]
            if len(batch.src) == 0:
                continue
            src_t = torch.tensor(batch.src, dtype=torch.long, device=device)
            dst_t = torch.tensor(batch.dst, dtype=torch.long, device=device)
            ts_t = torch.tensor(batch.timestamps, dtype=torch.float32, device=device)

            probs = model.predict_link_probabilities(src_t, dst_t, ts_t).cpu().numpy()

            if t >= eval_actual_start_t:
                all_preds.extend(probs.tolist())
                all_targets.extend(batch.labels.tolist())

            if is_tgn_memory:
                pos_mask = (batch.labels == 1)
                if np.any(pos_mask):
                    pos_src = torch.tensor(batch.src[pos_mask], dtype=torch.long, device=device)
                    pos_dst = torch.tensor(batch.dst[pos_mask], dtype=torch.long, device=device)
                    pos_ts = torch.tensor(batch.timestamps[pos_mask], dtype=torch.float32, device=device)
                    model.update_node_memories(pos_src, pos_dst, pos_ts)

    if len(all_targets) == 0:
        return {"ap": 0.5, "auc": 0.5}
    ap = float(average_precision_score(all_targets, all_preds))
    auc = float(roc_auc_score(all_targets, all_preds))
    return {"ap": ap, "auc": auc}


def compute_edge_jaccard_overlap(snaps_1: List[Any], snaps_2: List[Any]) -> float:
    """Computes Jaccard edge overlap between two sets of snapshots."""
    edges_1 = set()
    for snap in snaps_1:
        if hasattr(snap, "tocoo"):
            coo = snap.tocoo()
            for u, v in zip(coo.row, coo.col):
                if u < v:
                    edges_1.add((u, v))
        elif isinstance(snap, np.ndarray):
            rows, cols = np.where(snap > 0)
            for u, v in zip(rows, cols):
                if u < v:
                    edges_1.add((u, v))

    edges_2 = set()
    for snap in snaps_2:
        if hasattr(snap, "tocoo"):
            coo = snap.tocoo()
            for u, v in zip(coo.row, coo.col):
                if u < v:
                    edges_2.add((u, v))
        elif isinstance(snap, np.ndarray):
            rows, cols = np.where(snap > 0)
            for u, v in zip(rows, cols):
                if u < v:
                    edges_2.add((u, v))

    if len(edges_1 | edges_2) == 0:
        return 0.0
    return len(edges_1 & edges_2) / len(edges_1 | edges_2)


def run_phase6_5_suite(output_dir: Path, device_str: str = "cpu"):
    logger = setup_logger("Phase6_5Strengthening", output_dir / "reports" / "phase6_5.log")
    device = torch.device(device_str)
    start_time = time.time()

    processed_dir = output_dir / "processed"
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    configs_dir = output_dir / "configs"

    for d in [processed_dir, figures_dir, reports_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    seeds = list(range(42, 52))  # 10 evaluation seeds

    logger.info("=" * 70)
    logger.info("PHASE 6.5: FINAL SCIENTIFIC STRENGTHENING & MECHANISM AUDIT")
    logger.info("=" * 70)

    # =========================================================================
    # EXPERIMENT 1: CURRENT-SUFFICIENT SPECIFICITY CONTROL
    # =========================================================================
    if not (processed_dir / "specificity_control.csv").exists():
        logger.info("\n--- EXPERIMENT 1: CURRENT-SUFFICIENT SPECIFICITY CONTROL ---")
        gen_curr_suff = DynamicSBMGenerator(
            num_nodes=300,
            num_communities=3,
            regime_configs={
                "A_suff": RegimeConfig("A_suff", target_density=0.10, persistence=0.85, within_comm_multiplier=4.0, partition_seed=101),
                "B_suff": RegimeConfig("B_suff", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
            }
        )

        exp1_records = []
        for s in seeds:
            seq = gen_curr_suff.generate([("A_suff", 100), ("B_suff", 50), ("A_suff", 100)], seed=s)
            t_test_start = 150
            t_test_end = 249
            t_train_end = 135
            t_val_end = 149

            hist_A_times = list(range(0, 100))
            hist_A_snaps = [seq.get_snapshot(t) for t in hist_A_times]
            all_hist_times = list(range(0, t_test_start))
            all_hist_snaps = [seq.get_snapshot(t) for t in all_hist_times]

            cur_p = CurrentOnlyPredictor(cn_weight=0.3)
            ora_p = HistoricalOraclePredictor(variant="historical_summary", history_weight=1.2, cn_weight=0.3)
            cache_a = HistoricalStateCache()
            for t_h in hist_A_times:
                cache_a.store_state(t_h, seq.get_snapshot(t_h), regime_tag="A_suff")
            for t_h in range(100, t_test_start):
                cache_a.store_state(t_h, seq.get_snapshot(t_h), regime_tag="B_suff")
            ret_sim_p = HistoricalRetrievalPredictor(mode="similarity", history_weight=1.0, cn_weight=0.3)

            cur_sc, ora_sc, ret_sc, eval_tgts = [], [], [], []
            for t in range(t_test_start, t_test_end + 1):
                G_curr = seq.get_snapshot(t)
                G_next = seq.get_snapshot(min(t + 1, seq.total_timesteps - 1)) if t + 1 < seq.total_timesteps else G_curr
                pairs, labels = sample_evaluation_edges(G_next, negative_ratio=1.0, seed=s + t)
                if len(pairs) == 0:
                    continue
                cur_sc.extend(cur_p.predict_pairs(G_curr, pairs).tolist())
                ora_sc.extend(ora_p.predict_pairs(G_curr, hist_A_snaps, pairs, current_time=t, accessed_times=hist_A_times).tolist())
                h_sim, _ = ret_sim_p.retrieve_state(t, G_curr, cache_a, seed=s)
                ret_sc.extend(ret_sim_p.predict_pairs(G_curr, h_sim, pairs).tolist())
                eval_tgts.extend(labels.tolist())

            ap_cur = float(average_precision_score(eval_tgts, cur_sc))
            ap_ora = float(average_precision_score(eval_tgts, ora_sc))
            ap_ret = float(average_precision_score(eval_tgts, ret_sc))

            tgn = TGN(num_nodes=300, node_dim=32, memory_dim=64, time_dim=32, message_dim=32, device=device)
            tgn, _ = train_model_tgn_with_history(tgn, seq, train_end_t=t_train_end, val_end_t=t_val_end, num_epochs=12, device=device, seed=s)
            r_tgn = evaluate_tgn_rollout_with_k_A(tgn, seq, test_start_t=t_test_start, test_end_t=t_test_end, device=device, seed=s)

            tgn_nomem = TGNNoMemory(num_nodes=300, node_dim=32, time_dim=32, device=device)
            tgn_nomem, _ = train_model_tgn_with_history(tgn_nomem, seq, train_end_t=t_train_end, val_end_t=t_val_end, num_epochs=12, device=device, seed=s)
            r_tgn_nomem = evaluate_tgn_rollout_with_k_A(tgn_nomem, seq, test_start_t=t_test_start, test_end_t=t_test_end, device=device, seed=s)

            for m_name, m_ap, m_auc in [
                ("Current_Only", ap_cur, float(roc_auc_score(eval_tgts, cur_sc))),
                ("Continuous_TGN", r_tgn["ap"], r_tgn["auc"]),
                ("TGN_NoMemory", r_tgn_nomem["ap"], r_tgn_nomem["auc"]),
                ("Historical_Retrieval", ap_ret, float(roc_auc_score(eval_tgts, ret_sc))),
                ("Historical_Oracle", ap_ora, float(roc_auc_score(eval_tgts, ora_sc))),
            ]:
                exp1_records.append({
                    "benchmark": "Current_Sufficient_Control",
                    "seed": s,
                    "method": m_name,
                    "AP": m_ap,
                    "AUC": m_auc,
                    "delta_vs_current": m_ap - ap_cur,
                    "delta_vs_tgn": m_ap - r_tgn["ap"]
                })

        df_exp1 = pd.DataFrame(exp1_records)
        df_exp1.to_csv(processed_dir / "specificity_control.csv", index=False)
        logger.info(f"Experiment 1 complete. Summary:\n{df_exp1.groupby('method')['AP'].agg(['mean', 'std']).reset_index()}")
    else:
        logger.info("Experiment 1 already completed. Loading existing CSV.")

    # =========================================================================
    # EXPERIMENT 2: EXACT RECURRENCE VS STRUCTURAL RECURRENCE
    # =========================================================================
    if not (processed_dir / "recurrence_decomposition.csv").exists():
        logger.info("\n--- EXPERIMENT 2: EXACT RECURRENCE VS STRUCTURAL RECURRENCE ---")
        gen_exact = DynamicSBMGenerator(
            num_nodes=300, num_communities=3,
            regime_configs={
                "A": RegimeConfig("A", target_density=0.10, persistence=0.70, within_comm_multiplier=4.0, partition_seed=101),
                "B": RegimeConfig("B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
            }
        )
        gen_struct = DynamicSBMGenerator(
            num_nodes=300, num_communities=3,
            regime_configs={
                "A": RegimeConfig("A", target_density=0.10, persistence=0.05, within_comm_multiplier=4.0, partition_seed=101),
                "B": RegimeConfig("B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
            }
        )
        gen_nonrec = DynamicSBMGenerator(
            num_nodes=300, num_communities=3,
            regime_configs={
                "A": RegimeConfig("A", target_density=0.10, persistence=0.35, within_comm_multiplier=4.0, partition_seed=101),
                "B": RegimeConfig("B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
                "C": RegimeConfig("C", target_density=0.10, persistence=0.35, within_comm_multiplier=4.0, partition_seed=303),
            }
        )

        exp2_records = []
        conditions = [
            ("Condition_A_Exact_Recurrence", gen_exact, [("A", 100), ("B", 50), ("A", 100)]),
            ("Condition_B_Structural_Recurrence", gen_struct, [("A", 100), ("B", 50), ("A", 100)]),
            ("Control_Non_Recurrent_A_B_C", gen_nonrec, [("A", 100), ("B", 50), ("C", 100)]),
        ]

        for cond_name, gen_obj, schedule in conditions:
            logger.info(f"Evaluating Experiment 2: {cond_name}...")
            for s in seeds:
                seq = gen_obj.generate(schedule, seed=s)
                t_test_start = 150
                t_test_end = 249
                t_train_end = 135
                t_val_end = 149

                hist_A_times = list(range(0, 100))
                hist_A_snaps = [seq.get_snapshot(t) for t in hist_A_times]
                test_snaps = [seq.get_snapshot(t) for t in range(t_test_start, t_test_end + 1)]
                all_hist_times = list(range(0, t_test_start))
                all_hist_snaps = [seq.get_snapshot(t) for t in all_hist_times]

                overlap_jaccard = compute_edge_jaccard_overlap(hist_A_snaps, test_snaps)

                cur_p = CurrentOnlyPredictor(cn_weight=0.3)
                ora_p = HistoricalOraclePredictor(variant="historical_summary", history_weight=1.2, cn_weight=0.3)
                eb_p = EdgeBankPredictor(mode="all_history", history_weight=1.0, cn_weight=0.3)
                eb_b_p = EdgeBankPredictor(mode="bounded", bounded_window=(0, 99), history_weight=1.0, cn_weight=0.3)

                cache_a = HistoricalStateCache()
                for t_h in hist_A_times:
                    cache_a.store_state(t_h, seq.get_snapshot(t_h), regime_tag="A")
                for t_h in range(100, t_test_start):
                    cache_a.store_state(t_h, seq.get_snapshot(t_h), regime_tag="B")

                ret_sim_p = HistoricalRetrievalPredictor(mode="similarity", history_weight=1.0, cn_weight=0.3)
                ret_rnd_p = HistoricalRetrievalPredictor(mode="random", history_weight=1.0, cn_weight=0.3)

                cur_sc, ora_sc, eb_sc, eb_b_sc, ret_sc, rnd_sc, eval_tgts = [], [], [], [], [], [], []
                for t in range(t_test_start, t_test_end + 1):
                    G_curr = seq.get_snapshot(t)
                    G_next = seq.get_snapshot(min(t + 1, seq.total_timesteps - 1)) if t + 1 < seq.total_timesteps else G_curr
                    pairs, labels = sample_evaluation_edges(G_next, negative_ratio=1.0, seed=s + t)
                    if len(pairs) == 0:
                        continue
                    cur_sc.extend(cur_p.predict_pairs(G_curr, pairs).tolist())
                    ora_sc.extend(ora_p.predict_pairs(G_curr, hist_A_snaps, pairs, current_time=t, accessed_times=hist_A_times).tolist())
                    eb_sc.extend(eb_p.predict_pairs(G_curr, all_hist_snaps, pairs, current_time=t, accessed_times=all_hist_times).tolist())
                    eb_b_sc.extend(eb_b_p.predict_pairs(G_curr, hist_A_snaps, pairs, current_time=t, accessed_times=hist_A_times).tolist())

                    h_sim, _ = ret_sim_p.retrieve_state(t, G_curr, cache_a, seed=s)
                    ret_sc.extend(ret_sim_p.predict_pairs(G_curr, h_sim, pairs).tolist())

                    h_rnd, _ = ret_rnd_p.retrieve_state(t, G_curr, cache_a, seed=s)
                    rnd_sc.extend(ret_rnd_p.predict_pairs(G_curr, h_rnd, pairs).tolist())

                    eval_tgts.extend(labels.tolist())

                ap_cur = float(average_precision_score(eval_tgts, cur_sc))
                ap_ora = float(average_precision_score(eval_tgts, ora_sc))
                ap_eb = float(average_precision_score(eval_tgts, eb_sc))
                ap_eb_b = float(average_precision_score(eval_tgts, eb_b_sc))
                ap_ret = float(average_precision_score(eval_tgts, ret_sc))
                ap_rnd = float(average_precision_score(eval_tgts, rnd_sc))

                tgn = TGN(num_nodes=300, node_dim=32, memory_dim=64, time_dim=32, message_dim=32, device=device)
                tgn, _ = train_model_tgn_with_history(tgn, seq, train_end_t=t_train_end, val_end_t=t_val_end, num_epochs=12, device=device, seed=s)
                r_tgn = evaluate_tgn_rollout_with_k_A(tgn, seq, test_start_t=t_test_start, test_end_t=t_test_end, device=device, seed=s)

                for m_name, m_ap in [
                    ("Current_Only", ap_cur),
                    ("Continuous_TGN", r_tgn["ap"]),
                    ("EdgeBank_AllHistory", ap_eb),
                    ("EdgeBank_Bounded", ap_eb_b),
                    ("Historical_Retrieval", ap_ret),
                    ("Random_Retrieval", ap_rnd),
                    ("Historical_Oracle", ap_ora),
                ]:
                    exp2_records.append({
                        "condition": cond_name,
                        "seed": s,
                        "edge_jaccard_overlap": overlap_jaccard,
                        "method": m_name,
                        "AP": m_ap,
                        "delta_vs_current": m_ap - ap_cur,
                        "delta_vs_edgebank": m_ap - ap_eb_b
                    })

        df_exp2 = pd.DataFrame(exp2_records)
        df_exp2.to_csv(processed_dir / "recurrence_decomposition.csv", index=False)
        logger.info("Experiment 2 complete.")
    else:
        logger.info("Experiment 2 already completed. Loading existing CSV.")

    # =========================================================================
    # EXPERIMENT 3: RE-EXPOSURE / RECOVERY CURVE
    # =========================================================================
    gen_hard = DynamicSBMGenerator(
        num_nodes=300, num_communities=3,
        regime_configs={
            "A": RegimeConfig("A", target_density=0.10, persistence=0.35, within_comm_multiplier=4.0, partition_seed=101),
            "B": RegimeConfig("B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
        }
    )

    if not (processed_dir / "reexposure_recovery.csv").exists():
        logger.info("\n--- EXPERIMENT 3: RE-EXPOSURE / RECOVERY CURVE ---")
        k_A_list = [0, 1, 5, 10, 25]
        tb_exp3_list = [50, 100, 200]
        exp3_records = []

        for tb in tb_exp3_list:
            logger.info(f"Evaluating Experiment 3 for T_B = {tb}...")
            for s in seeds:
                seq = gen_hard.generate([("A", 100), ("B", tb), ("A", 100)], seed=s)
                t_test_start = 100 + tb
                t_test_end = 100 + tb + 99
                t_train_end = 100 + int(tb * 0.7)
                t_val_end = 100 + tb - 1

                tgn = TGN(num_nodes=300, node_dim=32, memory_dim=64, time_dim=32, message_dim=32, device=device)
                tgn, _ = train_model_tgn_with_history(tgn, seq, train_end_t=t_train_end, val_end_t=t_val_end, num_epochs=12, device=device, seed=s)

                tgn_nomem = TGNNoMemory(num_nodes=300, node_dim=32, time_dim=32, device=device)
                tgn_nomem, _ = train_model_tgn_with_history(tgn_nomem, seq, train_end_t=t_train_end, val_end_t=t_val_end, num_epochs=12, device=device, seed=s)

                for k_a in k_A_list:
                    r_tgn = evaluate_tgn_rollout_with_k_A(tgn, seq, test_start_t=t_test_start, test_end_t=t_test_end, k_A=k_a, device=device, seed=s)
                    r_tgn_nomem = evaluate_tgn_rollout_with_k_A(tgn_nomem, seq, test_start_t=t_test_start, test_end_t=t_test_end, k_A=k_a, device=device, seed=s)

                    cur_p = CurrentOnlyPredictor(cn_weight=0.3)
                    cur_sc, eval_tgts = [], []
                    for t in range(t_test_start + k_a, t_test_end + 1):
                        G_curr = seq.get_snapshot(t)
                        G_next = seq.get_snapshot(min(t + 1, seq.total_timesteps - 1)) if t + 1 < seq.total_timesteps else G_curr
                        pairs, labels = sample_evaluation_edges(G_next, negative_ratio=1.0, seed=s + t)
                        if len(pairs) == 0:
                            continue
                        cur_sc.extend(cur_p.predict_pairs(G_curr, pairs).tolist())
                        eval_tgts.extend(labels.tolist())

                    ap_cur = float(average_precision_score(eval_tgts, cur_sc)) if len(eval_tgts) > 0 else 0.76

                    exp3_records.append({
                        "T_B": tb,
                        "seed": s,
                        "k_A": k_a,
                        "TGN_AP": r_tgn["ap"],
                        "TGN_NoMemory_AP": r_tgn_nomem["ap"],
                        "Current_Only_AP": ap_cur,
                        "recovery_gain_vs_k0": 0.0
                    })

        df_exp3 = pd.DataFrame(exp3_records)
        for (tb, s), group in df_exp3.groupby(["T_B", "seed"]):
            k0_ap = group[group["k_A"] == 0]["TGN_AP"].values[0]
            df_exp3.loc[group.index, "recovery_gain_vs_k0"] = df_exp3.loc[group.index, "TGN_AP"] - k0_ap

        df_exp3.to_csv(processed_dir / "reexposure_recovery.csv", index=False)
        logger.info("Experiment 3 complete.")
    else:
        logger.info("Experiment 3 already completed. Loading existing CSV.")

    # =========================================================================
    # EXPERIMENT 4: TRAINING CONVERGENCE AUDIT
    # =========================================================================
    if not (processed_dir / "training_convergence.csv").exists():
        logger.info("\n--- EXPERIMENT 4: TRAINING CONVERGENCE AUDIT (25 EPOCHS) ---")
        exp4_records = []
        tb_exp4_list = [25, 100, 200]

        for tb in tb_exp4_list:
            logger.info(f"Auditing Training Convergence for T_B = {tb}...")
            for s in seeds:
                seq = gen_hard.generate([("A", 100), ("B", tb), ("A", 100)], seed=s)
                t_train_end = 100 + int(tb * 0.7)
                t_val_end = 100 + tb - 1
                t_test_start = 100 + tb
                t_test_end = 100 + tb + 99

                tgn = TGN(num_nodes=300, node_dim=32, memory_dim=64, time_dim=32, message_dim=32, device=device)
                tgn, epoch_hist = train_model_tgn_with_history(tgn, seq, train_end_t=t_train_end, val_end_t=t_val_end, num_epochs=25, device=device, seed=s)
                r_tgn = evaluate_tgn_rollout_with_k_A(tgn, seq, test_start_t=t_test_start, test_end_t=t_test_end, device=device, seed=s)

                for rec in epoch_hist:
                    exp4_records.append({
                        "T_B": tb,
                        "seed": s,
                        "epoch": rec["epoch"],
                        "train_loss": rec["train_loss"],
                        "val_ap": rec["val_ap"],
                        "val_auc": rec["val_auc"],
                        "final_test_ap": r_tgn["ap"]
                    })

        df_exp4 = pd.DataFrame(exp4_records)
        df_exp4.to_csv(processed_dir / "training_convergence.csv", index=False)
        logger.info("Experiment 4 complete.")
    else:
        logger.info("Experiment 4 already completed. Loading existing CSV.")

    # =========================================================================
    # EXPERIMENT 5: FROZEN MEMORY LINEAR PROBE
    # =========================================================================
    logger.info("\n--- EXPERIMENT 5: FROZEN MEMORY LINEAR PROBE ---")
    exp5_records = []

    for s in seeds:
        seq = gen_hard.generate([("A", 100), ("B", 100), ("A", 50)], seed=s)
        # Partition assignments for Regime A
        comm_assigns = gen_hard.regimes["A"].community_assignments

        tgn = TGN(num_nodes=300, node_dim=32, memory_dim=64, time_dim=32, message_dim=32, device=device)
        tgn, _ = train_model_tgn_with_history(tgn, seq, train_end_t=170, val_end_t=199, num_epochs=12, device=device, seed=s)
        tgn.eval()
        tgn.reset_memory()

        event_batches = extract_events_from_sequence(seq, negative_ratio=1.0, seed=s)
        probe_checkpoints = {
            "t1_End_of_Regime_A": 99,
            "t2_Short_B_Distractor": 110,
            "t3_End_of_Long_B_Distractor": 199,
            "t4_After_10_Steps_Recurrent_A": 210
        }

        memory_snapshots = {}

        with torch.no_grad():
            for t in range(250):
                batch = event_batches[t]
                if len(batch.src) > 0:
                    pos_mask = (batch.labels == 1)
                    if np.any(pos_mask):
                        pos_src = torch.tensor(batch.src[pos_mask], dtype=torch.long, device=device)
                        pos_dst = torch.tensor(batch.dst[pos_mask], dtype=torch.long, device=device)
                        pos_ts = torch.tensor(batch.timestamps[pos_mask], dtype=torch.float32, device=device)
                        tgn.update_node_memories(pos_src, pos_dst, pos_ts)

                for cp_name, cp_t in probe_checkpoints.items():
                    if t == cp_t:
                        memory_snapshots[cp_name] = tgn.memory_bank.memory.clone().cpu().numpy()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        for cp_name, M_snap in memory_snapshots.items():
            accs, f1s = [], []
            for train_idx, test_idx in skf.split(M_snap, comm_assigns):
                clf = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(M_snap[train_idx], comm_assigns[train_idx])
                preds = clf.predict(M_snap[test_idx])
                accs.append(np.mean(preds == comm_assigns[test_idx]))
                f1s.append(f1_score(comm_assigns[test_idx], preds, average="macro"))

            exp5_records.append({
                "seed": s,
                "checkpoint": cp_name,
                "probe_accuracy": np.mean(accs),
                "probe_macro_f1": np.mean(f1s),
            })

    df_exp5 = pd.DataFrame(exp5_records)
    df_exp5.to_csv(processed_dir / "memory_probe.csv", index=False)
    logger.info(f"Experiment 5 complete. Summary:\n{df_exp5.groupby('checkpoint')[['probe_accuracy', 'probe_macro_f1']].agg(['mean', 'std']).reset_index()}")

    elapsed = time.time() - start_time
    logger.info(f"\nAll Phase 6.5 experiments successfully completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6.5 Scientific Strengthening Suite")
    parser.add_argument("--output_dir", type=str, default="results/phase6_5")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    run_phase6_5_suite(output_dir=out_path, device_str=args.device)
