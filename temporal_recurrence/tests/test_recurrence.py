"""Unit tests for recurrence scheduling, oracle indexing, and anti-leakage checks."""
import unittest
import numpy as np

from src.generator.dsbm import DynamicSBMGenerator, RegimeScheduler
from src.evaluation.prediction import verify_no_future_leakage
from src.baselines.historical_oracle import (
    HistoricalOraclePredictor,
    TrueRegimeOraclePredictor,
    RandomHistoryControlPredictor
)


class TestRecurrence(unittest.TestCase):

    def setUp(self):
        self.spec = [("A", 50), ("B", 100), ("A", 50)]
        self.scheduler = RegimeScheduler(self.spec)
        self.generator = DynamicSBMGenerator(num_nodes=60, num_communities=3)

    def test_schedule_structure(self):
        self.assertEqual(self.scheduler.total_timesteps, 200)
        self.assertEqual(len(self.scheduler.episodes), 3)
        self.assertEqual(self.scheduler.episodes[0]["regime"], "A")
        self.assertEqual(self.scheduler.episodes[1]["regime"], "B")
        self.assertEqual(self.scheduler.episodes[2]["regime"], "A")

    def test_previous_episode_lookup(self):
        # At t=150 (inside second A), previous A episode should be [0, 49]
        prev_A = self.scheduler.get_previous_episodes_for_regime("A", before_time=150)
        self.assertEqual(len(prev_A), 1)
        self.assertEqual(prev_A[0]["start"], 0)
        self.assertEqual(prev_A[0]["end"], 49)

    def test_anti_leakage_assertion(self):
        # Accessing past timestamps is allowed
        try:
            verify_no_future_leakage(current_time=150, accessed_times=[0, 10, 49, 140, 150])
        except ValueError:
            self.fail("verify_no_future_leakage raised ValueError unexpectedly on valid past timestamps!")

        # Accessing future timestamp must raise ValueError
        with self.assertRaises(ValueError):
            verify_no_future_leakage(current_time=150, accessed_times=[10, 151])

    def test_oracle_leakage_guard(self):
        seq = self.generator.generate(self.spec, seed=42)
        oracle = HistoricalOraclePredictor()
        pairs = np.array([[0, 1], [2, 3]])
        G_t = seq.get_snapshot(150)
        hist_snaps = [seq.get_snapshot(0)]

        # Future accessed times should throw error
        with self.assertRaises(ValueError):
            oracle.predict_pairs(
                G_t, hist_snaps, pairs, current_time=150, accessed_times=[151]
            )

    def test_random_history_control(self):
        seq = self.generator.generate(self.spec, seed=42)
        random_control = RandomHistoryControlPredictor()
        pairs = np.array([[0, 1], [2, 3]])
        G_t = seq.get_snapshot(150)
        rand_snaps = [seq.get_snapshot(10)]
        scores = random_control.predict_pairs(
            G_t, rand_snaps, pairs, current_time=150, accessed_times=[10]
        )
        self.assertEqual(len(scores), 2)


if __name__ == "__main__":
    unittest.main()
