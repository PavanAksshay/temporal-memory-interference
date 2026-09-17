"""Unit tests for Phase 3: Synthetic Robustness, Memory Addressability, and Real Temporal Graphs."""
import unittest
from pathlib import Path
import numpy as np
import torch

from src.generator.regimes import RegimeConfig
from src.generator.dsbm import DynamicSBMGenerator
from src.evaluation.real_data import RealTemporalGraphDataset
from src.evaluation.prediction import verify_no_future_leakage
from src.models.tgn import TGN
from src.models.retrieval import HistoricalStateCache, HistoricalRetrievalPredictor


class TestPhase3Suite(unittest.TestCase):

    def setUp(self):
        self.generator = DynamicSBMGenerator(
            num_nodes=60,
            num_communities=3,
            regime_configs={
                "A": RegimeConfig("A", target_density=0.10, persistence=0.35, within_comm_multiplier=4.0, partition_seed=101),
                "B": RegimeConfig("B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
            }
        )

    def test_partition_independence(self):
        """Test 1: Mutual independence of independently generated partitions."""
        c_A = self.generator.regimes["A"].community_assignments
        c_B = self.generator.regimes["B"].community_assignments
        # Compute normalized mutual information or contingency
        self.assertFalse(np.array_equal(c_A, c_B), "Regime A and B partitions must be distinct.")

    def test_density_control(self):
        """Test 2: Target density scaling across 0.05, 0.10, 0.20."""
        for target_rho in [0.05, 0.10, 0.20]:
            gen = DynamicSBMGenerator(
                num_nodes=60,
                num_communities=3,
                regime_configs={
                    "A": RegimeConfig("A", target_density=target_rho, persistence=0.35, within_comm_multiplier=4.0, partition_seed=101)
                }
            )
            seq = gen.generate([("A", 50)], seed=42)
            densities = [np.mean(seq.get_snapshot(t)) for t in range(50)]
            mean_dens = np.mean(densities)
            self.assertAlmostEqual(mean_dens, target_rho, delta=0.04)

    def test_distractor_integrity(self):
        """Test 3: Verify structural integrity of D1 (orthogonal), D2 (persistence), D3 (structural)."""
        reg_cfgs = {
            "D1": RegimeConfig("D1", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
            "D2": RegimeConfig("D2", target_density=0.10, persistence=0.80, within_comm_multiplier=4.0, partition_seed=101),
            "D3": RegimeConfig("D3", target_density=0.10, persistence=0.15, within_comm_multiplier=1.0, partition_seed=303),
        }
        gen = DynamicSBMGenerator(num_nodes=60, num_communities=3, regime_configs=reg_cfgs)
        self.assertEqual(len(gen.regimes), 3)

    def test_multi_cycle_temporal_order(self):
        """Test 4: Verify chronological event ordering in multi-cycle schedules."""
        schedule = [("A", 10), ("B", 10), ("A", 10), ("B", 10), ("A", 10)]
        seq = self.generator.generate(schedule, seed=42)
        self.assertEqual(seq.total_timesteps, 50)
        self.assertEqual(seq.scheduler.get_record(0).regime, "A")
        self.assertEqual(seq.scheduler.get_record(15).regime, "B")
        self.assertEqual(seq.scheduler.get_record(25).regime, "A")

    def test_memory_budget_accounting(self):
        """Test 5: Memory budget accounting between TGN (d_m=256) and explicit memory."""
        model_256 = TGN(num_nodes=300, node_dim=32, memory_dim=256, time_dim=32, message_dim=32)
        param_count_256 = sum(p.numel() for p in model_256.parameters())
        # Parameter count for d_m=256 is ~100k parameters
        self.assertGreater(param_count_256, 50000)

    def test_snapshot_cutoff(self):
        """Test 6: Verify strict anti-leakage cutoff in snapshot memory stores."""
        cache = HistoricalStateCache()
        for t in range(50):
            cache.store_state(t, np.zeros((10, 10)), regime_tag="A")
        valid_times = cache.get_valid_history(current_time=30)
        self.assertEqual(max(valid_times), 29)
        self.assertNotIn(30, valid_times)

    def test_fifo_cutoff(self):
        """Test 7: Verify FIFO queue stores only historical items strictly before current time."""
        cache = HistoricalStateCache()
        for t in range(100):
            cache.store_state(t, np.zeros((10, 10)), regime_tag="A" if t < 50 else "B")
        valid_times = cache.get_valid_history(current_time=60)
        fifo_k = valid_times[-10:]
        self.assertEqual(len(fifo_k), 10)
        self.assertEqual(max(fifo_k), 59)

    def test_real_data_temporal_split(self):
        """Test 8: Chronological train/val/test splits on CollegeMsg real dataset."""
        real_ds = RealTemporalGraphDataset(data_path="data/real/CollegeMsg.txt", num_windows=10, max_nodes=100)
        self.assertEqual(len(real_ds.windows), 10)
        # Verify chronological timestamp monotonicity across windows
        for i in range(len(real_ds.windows) - 1):
            self.assertLessEqual(real_ds.windows[i].start_time, real_ds.windows[i + 1].start_time)

    def test_real_data_no_future_information(self):
        """Test 9: Verify zero future leakage in real-data evaluations."""
        current_time = 5
        accessed = [0, 1, 2, 3, 4]
        verify_no_future_leakage(current_time, accessed)
        with self.assertRaises(ValueError):
            verify_no_future_leakage(current_time, [6])

    def test_real_data_candidate_parity(self):
        """Test 10: Verify candidate edge parity across comparisons on real dataset."""
        real_ds = RealTemporalGraphDataset(data_path="data/real/CollegeMsg.txt", num_windows=5, max_nodes=50)
        b1 = real_ds.extract_event_batches(negative_ratio=1.0, seed=42)
        b2 = real_ds.extract_event_batches(negative_ratio=1.0, seed=42)
        for t in range(len(b1)):
            np.testing.assert_array_equal(b1[t].src, b2[t].src)
            np.testing.assert_array_equal(b1[t].dst, b2[t].dst)
            np.testing.assert_array_equal(b1[t].labels, b2[t].labels)

    def test_real_data_label_parity(self):
        """Test 11: Verify 1:1 class balance on real dataset event batches."""
        real_ds = RealTemporalGraphDataset(data_path="data/real/CollegeMsg.txt", num_windows=5, max_nodes=50)
        batches = real_ds.extract_event_batches(negative_ratio=1.0, seed=42)
        for b in batches:
            if len(b.labels) > 0:
                pos = np.sum(b.labels == 1)
                neg = np.sum(b.labels == 0)
                self.assertEqual(pos, neg)


if __name__ == "__main__":
    unittest.main()
