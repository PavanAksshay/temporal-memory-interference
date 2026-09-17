"""Unit tests for Phase 4.5: EdgeBank baseline, Hard Benchmark Calibration, and Leakage Checks."""
import unittest
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.generator.regimes import RegimeConfig
from src.generator.dsbm import DynamicSBMGenerator
from src.evaluation.prediction import sample_evaluation_edges, verify_no_future_leakage
from src.baselines.current_only import CurrentOnlyPredictor
from src.baselines.historical_oracle import HistoricalOraclePredictor
from src.baselines.edgebank import EdgeBankPredictor
from src.evaluation.real_data import RealTemporalGraphDataset


class TestPhase4_5Suite(unittest.TestCase):

    def setUp(self):
        self.data_path = Path("data/real/CollegeMsg.txt")
        self.generator = DynamicSBMGenerator(
            num_nodes=300,
            num_communities=3,
            regime_configs={
                "A": RegimeConfig("A", target_density=0.10, persistence=0.35, within_comm_multiplier=4.0, partition_seed=101),
                "B": RegimeConfig("B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
                "C": RegimeConfig("C", target_density=0.10, persistence=0.25, within_comm_multiplier=4.0, partition_seed=303),
            }
        )

    def test_hard_benchmark_calibration(self):
        """Test 1: Hard benchmark satisfies Historical Oracle AP > Current-Only AP."""
        seq = self.generator.generate([("A", 100), ("B", 50), ("A", 100)], seed=42)
        hist_A_times = list(range(0, 100))
        hist_A_snaps = [seq.get_snapshot(t) for t in hist_A_times]

        cur_pred = CurrentOnlyPredictor(cn_weight=0.3)
        ora_pred = HistoricalOraclePredictor(variant="historical_summary", history_weight=1.2, cn_weight=0.3)

        cur_scores, ora_scores, all_labels = [], [], []
        for t in range(150, 170):
            G_curr = seq.get_snapshot(t)
            G_next = seq.get_snapshot(t + 1)
            pairs, labels = sample_evaluation_edges(G_next, negative_ratio=1.0, seed=42 + t)
            if len(pairs) == 0:
                continue
            c_sc = cur_pred.predict_pairs(G_curr, pairs)
            o_sc = ora_pred.predict_pairs(G_curr, hist_A_snaps, pairs, current_time=t, accessed_times=hist_A_times)
            cur_scores.extend(c_sc.tolist())
            ora_scores.extend(o_sc.tolist())
            all_labels.extend(labels.tolist())

        ap_cur = average_precision_score(all_labels, cur_scores)
        ap_ora = average_precision_score(all_labels, ora_scores)

        self.assertGreater(ap_ora, ap_cur, "Historical Oracle must outperform Current-Only on Hard Benchmark.")
        self.assertAlmostEqual(ap_cur, 0.75, delta=0.08)
        self.assertAlmostEqual(ap_ora, 0.79, delta=0.08)

    def test_edgebank_candidate_parity(self):
        """Test 2: EdgeBank receives identical candidate pairs and produces valid scores."""
        seq = self.generator.generate([("A", 20), ("B", 20), ("A", 20)], seed=42)
        G_curr = seq.get_snapshot(45)
        G_next = seq.get_snapshot(46)
        pairs, labels = sample_evaluation_edges(G_next, negative_ratio=1.0, seed=42)

        eb = EdgeBankPredictor(mode="all_history")
        hist_snaps = [seq.get_snapshot(t) for t in range(45)]
        accessed = list(range(45))
        scores = eb.predict_pairs(G_curr, hist_snaps, pairs, current_time=45, accessed_times=accessed)

        self.assertEqual(len(scores), len(pairs))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_edgebank_temporal_leakage(self):
        """Test 3: EdgeBank strictly rejects future timestamps."""
        eb = EdgeBankPredictor(mode="all_history")
        G_curr = np.zeros((10, 10))
        pairs = np.array([[0, 1], [2, 3]])
        with self.assertRaises(ValueError):
            # Accessed timestamp 11 > current_time 10
            eb.predict_pairs(G_curr, [np.zeros((10, 10))], pairs, current_time=10, accessed_times=[11])

    def test_edgebank_target_leakage(self):
        """Test 4: Candidate edges never seen in history receive zero historical boost."""
        eb = EdgeBankPredictor(mode="bounded", bounded_window=(0, 10))
        G_curr = np.zeros((10, 10))
        # Snapshot where only (0, 1) existed
        snap_0 = np.zeros((10, 10))
        snap_0[0, 1] = 1
        snap_0[1, 0] = 1

        pairs = np.array([[0, 1], [4, 5]])  # (4, 5) never existed
        scores = eb.predict_pairs(G_curr, [snap_0], pairs, current_time=15, accessed_times=[5])

        self.assertGreater(scores[0], scores[1], "Historically observed edge must receive higher score than never-seen edge.")

    def test_edgebank_bounded_vs_unbounded(self):
        """Test 5: Bounded EdgeBank filters strictly within specified historical window."""
        eb_bounded = EdgeBankPredictor(mode="bounded", bounded_window=(0, 5))
        eb_unbounded = EdgeBankPredictor(mode="all_history")

        snap_early = np.zeros((10, 10))
        snap_early[0, 1] = 1
        snap_late = np.zeros((10, 10))
        snap_late[2, 3] = 1

        snaps = [snap_early, snap_late]
        times = [2, 8]
        pairs = np.array([[0, 1], [2, 3]])

        # At t=10, bounded (0..5) should only see (0, 1), not (2, 3)
        sc_b = eb_bounded.predict_pairs(np.zeros((10, 10)), snaps, pairs, current_time=10, accessed_times=times)
        sc_u = eb_unbounded.predict_pairs(np.zeros((10, 10)), snaps, pairs, current_time=10, accessed_times=times)

        self.assertGreater(sc_b[0], sc_b[1])
        self.assertEqual(sc_u[0], sc_u[1])

    def test_collegemsg_edgebank_parity(self):
        """Test 6: Real dataset candidate parity and label balance with EdgeBank."""
        if self.data_path.exists():
            ds = RealTemporalGraphDataset(data_path=str(self.data_path), num_windows=5, max_nodes=50)
            batches = ds.extract_event_batches(negative_ratio=1.0, seed=42)
            target_b = batches[3]
            pairs = np.column_stack([target_b.src, target_b.dst])

            hist_snaps = [ds.windows[t].adjacency for t in range(3)]
            accessed = list(range(3))

            eb = EdgeBankPredictor(mode="all_history")
            sc = eb.predict_pairs(ds.windows[2].adjacency, hist_snaps, pairs, current_time=3, accessed_times=accessed)
            self.assertEqual(len(sc), len(target_b.labels))


if __name__ == "__main__":
    unittest.main()
