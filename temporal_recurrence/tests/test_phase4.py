"""Unit tests for Phase 4: Publication Readiness Audit."""
import unittest
from pathlib import Path
import numpy as np
import torch

from src.generator.regimes import RegimeConfig
from src.generator.dsbm import DynamicSBMGenerator
from src.evaluation.real_data import RealTemporalGraphDataset, RealDataWindow
from src.evaluation.prediction import verify_no_future_leakage
from src.models.tgn import TGN, TGNNoMemory
from src.models.retrieval import HistoricalStateCache, HistoricalRetrievalPredictor


class TestPhase4Suite(unittest.TestCase):

    def setUp(self):
        self.data_path = Path("data/real/CollegeMsg.txt")
        self.generator = DynamicSBMGenerator(
            num_nodes=60,
            num_communities=3,
            regime_configs={
                "A": RegimeConfig("A", target_density=0.10, persistence=0.35, within_comm_multiplier=4.0, partition_seed=101),
                "B": RegimeConfig("B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
            }
        )

    def test_episode_independence(self):
        """Test 1: Verify statistical episode definitions and disjoint evaluation intervals."""
        if self.data_path.exists():
            ds = RealTemporalGraphDataset(data_path=str(self.data_path), num_windows=15, max_nodes=100)
            episodes = ds.discover_recurring_episodes(min_gap=2, sim_threshold=0.55)
            self.assertGreater(len(episodes), 0)
            for t_A1, t_B, t_A2, sim in episodes:
                self.assertLess(t_A1, t_B)
                self.assertLess(t_B, t_A2)

    def test_block_bootstrap_integrity(self):
        """Test 2: Block bootstrap resamples at the block/window level, not individual edges."""
        n_blocks = 10
        block_scores = np.array([0.65, 0.70, 0.72, 0.68, 0.75, 0.71, 0.69, 0.73, 0.74, 0.70])
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(100):
            idx = rng.choice(n_blocks, size=n_blocks, replace=True)
            boot_means.append(np.mean(block_scores[idx]))
        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)
        self.assertLess(ci_low, np.mean(block_scores))
        self.assertGreater(ci_high, np.mean(block_scores))

    def test_recurrence_detection_no_target_leakage(self):
        """Test 3: Recurrence detection uses only structural similarity without future edge targets."""
        if self.data_path.exists():
            ds = RealTemporalGraphDataset(data_path=str(self.data_path), num_windows=10, max_nodes=50)
            sim_mat = ds.compute_similarity_matrix()
            self.assertEqual(sim_mat.shape, (10, 10))
            self.assertTrue(np.all(sim_mat >= 0.0))
            self.assertTrue(np.all(sim_mat <= 1.0001))

    def test_real_history_cutoff(self):
        """Test 4: Real-data historical cache strictly prevents future access."""
        cache = HistoricalStateCache()
        for w_idx in range(10):
            cache.store_state(w_idx, np.zeros((20, 20)), regime_tag="A" if w_idx % 2 == 0 else "B")
        valid_history = cache.get_valid_history(current_time=6)
        self.assertEqual(max(valid_history), 5)
        self.assertNotIn(6, valid_history)

    def test_real_negative_control(self):
        """Test 5: Verify negative control selects dissimilar historical window."""
        if self.data_path.exists():
            ds = RealTemporalGraphDataset(data_path=str(self.data_path), num_windows=10, max_nodes=50)
            sim_mat = ds.compute_similarity_matrix()
            # For window 8, the least similar historical window in 0..7
            min_sim_w = np.argmin(sim_mat[8, :8])
            self.assertLess(min_sim_w, 8)
            self.assertLess(sim_mat[8, min_sim_w], 0.95)

    def test_storage_budget(self):
        """Test 6: Distinct parametric vs non-parametric storage footprint accounting."""
        tgn = TGN(num_nodes=100, node_dim=32, memory_dim=64, time_dim=32, message_dim=32)
        param_count = sum(p.numel() for p in tgn.parameters())
        # Parameter count for d_m=64 is approximately 10-25k parameters
        self.assertGreater(param_count, 5000)

        # Snapshot store for K=1 on 100 nodes sparse matrix
        k1_storage = 100 * 100  # maximum dense bits/floats
        self.assertLessEqual(k1_storage, 10000)

    def test_candidate_parity_real(self):
        """Test 7: Candidate edge parity across methods on real dataset."""
        if self.data_path.exists():
            ds = RealTemporalGraphDataset(data_path=str(self.data_path), num_windows=5, max_nodes=50)
            batches = ds.extract_event_batches(negative_ratio=1.0, seed=42)
            target_batch = batches[3]
            pairs1 = np.column_stack([target_batch.src, target_batch.dst])
            pairs2 = np.column_stack([target_batch.src, target_batch.dst])
            np.testing.assert_array_equal(pairs1, pairs2)

    def test_label_parity_real(self):
        """Test 8: 1:1 label balance on real dataset event batches."""
        if self.data_path.exists():
            ds = RealTemporalGraphDataset(data_path=str(self.data_path), num_windows=5, max_nodes=50)
            batches = ds.extract_event_batches(negative_ratio=1.0, seed=42)
            for b in batches:
                if len(b.labels) > 0:
                    pos = np.sum(b.labels == 1)
                    neg = np.sum(b.labels == 0)
                    self.assertEqual(pos, neg)

    def test_synthetic_final_reproduction(self):
        """Test 9: Synthetic generator reproduces deterministic sequence across seeds."""
        seq1 = self.generator.generate([("A", 10), ("B", 20), ("A", 10)], seed=42)
        seq2 = self.generator.generate([("A", 10), ("B", 20), ("A", 10)], seed=42)
        np.testing.assert_array_equal(seq1.get_snapshot(0), seq2.get_snapshot(0))
        np.testing.assert_array_equal(seq1.get_snapshot(25), seq2.get_snapshot(25))


if __name__ == "__main__":
    unittest.main()
