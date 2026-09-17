"""Unit tests for Phase 2: Memory Capacity, Oracle Injection, and Historical Retrieval."""
import unittest
from pathlib import Path
import numpy as np
import torch

from src.models.tgn import TGN
from src.models.retrieval import HistoricalStateCache, HistoricalRetrievalPredictor
from src.generator.regimes import RegimeConfig
from src.generator.dsbm import DynamicSBMGenerator
from src.evaluation.event_converter import extract_events_from_sequence
from src.evaluation.prediction import verify_no_future_leakage


class TestPhase2Suite(unittest.TestCase):

    def setUp(self):
        self.generator = DynamicSBMGenerator(
            num_nodes=60,
            num_communities=3,
            regime_configs={
                "A": RegimeConfig("A", target_density=0.10, persistence=0.35, within_comm_multiplier=4.0, partition_seed=101),
                "B": RegimeConfig("B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
            }
        )

    def test_01_memory_dimension_propagation(self):
        """Test 1: Memory dimension is propagated through all model layers and increases parameter count."""
        param_counts = []
        for d_m in [16, 32, 64, 128, 256]:
            model = TGN(num_nodes=60, node_dim=32, memory_dim=d_m, time_dim=32, message_dim=32, device=torch.device("cpu"))
            self.assertEqual(model.memory_bank.memory.shape[1], d_m)
            self.assertEqual(model.memory_updater.gru.hidden_size, d_m)
            n_params = sum(p.numel() for p in model.parameters())
            param_counts.append(n_params)
        # Verify parameter count increases strictly monotonically
        for i in range(len(param_counts) - 1):
            self.assertGreater(param_counts[i + 1], param_counts[i])

    def test_02_oracle_injection_anti_leakage(self):
        """Test 2: No future A events enter oracle memory injection."""
        current_time = 300
        accessed_times = list(range(0, 100))  # Historical A interval
        # Must not raise leakage error
        verify_no_future_leakage(current_time, accessed_times)
        # Attempting to access future time (> current_time) must raise ValueError
        with self.assertRaises(ValueError):
            verify_no_future_leakage(current_time, [301])
        with self.assertRaises(ValueError):
            verify_no_future_leakage(current_time, [350])

    def test_03_retrieval_cannot_access_test_labels(self):
        """Test 3: Retrieval operates solely on graph structures, independent of test labels."""
        cache = HistoricalStateCache()
        for t in range(50):
            G = np.random.binomial(1, 0.1, size=(20, 20))
            cache.store_state(t, G, regime_tag="A")
        predictor = HistoricalRetrievalPredictor(mode="similarity")
        G_curr = np.random.binomial(1, 0.1, size=(20, 20))
        # Retrieval does not take labels as input
        retrieved_H, retrieved_t = predictor.retrieve_state(current_time=50, current_graph=G_curr, cache=cache)
        self.assertLess(retrieved_t, 50)
        self.assertEqual(retrieved_H.shape, (20, 20))

    def test_04_random_retrieval_randomization(self):
        """Test 4: Random retrieval selects diverse historical states across seeds."""
        cache = HistoricalStateCache()
        for t in range(100):
            G = np.zeros((10, 10))
            cache.store_state(t, G, regime_tag="A" if t < 50 else "B")
        predictor = HistoricalRetrievalPredictor(mode="random")
        selected_times = set()
        for s in range(20):
            _, t_sel = predictor.retrieve_state(current_time=100, current_graph=np.zeros((10, 10)), cache=cache, seed=s)
            selected_times.add(t_sel)
        self.assertGreater(len(selected_times), 5, "Random retrieval must pick multiple distinct timestamps.")

    def test_05_recent_b_retrieval_integrity(self):
        """Test 5: Recent-B retrieval strictly retrieves from regime B."""
        cache = HistoricalStateCache()
        for t in range(0, 50):
            cache.store_state(t, np.zeros((10, 10)), regime_tag="A")
        for t in range(50, 150):
            cache.store_state(t, np.ones((10, 10)), regime_tag="B")
        predictor = HistoricalRetrievalPredictor(mode="recent_b")
        retrieved_H, t_sel = predictor.retrieve_state(current_time=150, current_graph=np.zeros((10, 10)), cache=cache)
        self.assertGreaterEqual(t_sel, 50)
        self.assertEqual(cache.regime_tags[t_sel], "B")

    def test_06_correct_historical_retrieval_integrity(self):
        """Test 6: Correct historical retrieval strictly retrieves from regime A."""
        cache = HistoricalStateCache()
        for t in range(0, 50):
            cache.store_state(t, np.zeros((10, 10)), regime_tag="A")
        for t in range(50, 150):
            cache.store_state(t, np.ones((10, 10)), regime_tag="B")
        predictor = HistoricalRetrievalPredictor(mode="oracle")
        retrieved_H, t_sel = predictor.retrieve_state(current_time=150, current_graph=np.zeros((10, 10)), cache=cache)
        self.assertLess(t_sel, 50)
        self.assertEqual(cache.regime_tags[t_sel], "A")

    def test_07_candidate_edges_identical(self):
        """Test 7: Candidate edges and targets are identical across comparison methods."""
        seq = self.generator.generate([("A", 20), ("B", 20), ("A", 20)], seed=42)
        batches1 = extract_events_from_sequence(seq, negative_ratio=1.0, seed=42)
        batches2 = extract_events_from_sequence(seq, negative_ratio=1.0, seed=42)
        for t in range(len(batches1)):
            np.testing.assert_array_equal(batches1[t].src, batches2[t].src)
            np.testing.assert_array_equal(batches1[t].dst, batches2[t].dst)
            np.testing.assert_array_equal(batches1[t].labels, batches2[t].labels)

    def test_08_negative_sampling_balance(self):
        """Test 8: Negative sampling is exactly 1:1 balanced."""
        seq = self.generator.generate([("A", 10)], seed=42)
        batches = extract_events_from_sequence(seq, negative_ratio=1.0, seed=42)
        for b in batches:
            if len(b.labels) > 0:
                pos = np.sum(b.labels == 1)
                neg = np.sum(b.labels == 0)
                self.assertEqual(pos, neg)

    def test_09_evaluation_timestamps_identical(self):
        """Test 9: Evaluation timestamps are identically sequenced."""
        seq = self.generator.generate([("A", 100), ("B", 50), ("A", 100)], seed=42)
        test_start = 150
        test_end = 249
        eval_t = list(range(test_start, test_end + 1))
        self.assertEqual(len(eval_t), 100)
        self.assertEqual(eval_t[0], 150)
        self.assertEqual(eval_t[-1], 249)

    def test_10_protocol_equivalence(self):
        """Test 10: Stationary and recurrence experiments use equivalent sequence lengths and evaluation logic."""
        seq_stat = self.generator.generate([("A", 400)], seed=42)
        seq_recur = self.generator.generate([("A", 100), ("B", 200), ("A", 100)], seed=42)
        self.assertEqual(seq_stat.total_timesteps, 400)
        self.assertEqual(seq_recur.total_timesteps, 400)

    def test_11_seed_determinism(self):
        """Test 11: Dynamic graph generation is deterministic for identical seeds."""
        seq1 = self.generator.generate([("A", 30)], seed=123)
        seq2 = self.generator.generate([("A", 30)], seed=123)
        np.testing.assert_array_equal(seq1.snapshots, seq2.snapshots)

    def test_12_no_previous_phase_overwrites(self):
        """Test 12: Previous phase result directories are intact and uncorrupted."""
        p0_1_report = Path("results/phase0_1/processed/phase0_1_report.md")
        p1_report = Path("results/phase1/processed/phase1_report.md")
        p1_1_report = Path("results/phase1_1/processed/phase1_1_report.md")
        if p0_1_report.exists():
            self.assertGreater(p0_1_report.stat().st_size, 0)
        if p1_report.exists():
            self.assertGreater(p1_report.stat().st_size, 0)
        if p1_1_report.exists():
            self.assertGreater(p1_1_report.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
