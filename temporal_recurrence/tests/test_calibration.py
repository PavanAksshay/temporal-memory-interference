"""Unit tests for regime calibration and similarity."""
import unittest
import numpy as np

from src.generator.dsbm import DynamicSBMGenerator
from src.generator.calibration import RegimeCalibrator
from src.generator.regimes import RegimeConfig


class TestCalibration(unittest.TestCase):

    def setUp(self):
        self.regime_configs = {
            "A": RegimeConfig("A", target_density=0.10, persistence=0.35, within_comm_multiplier=4.0, partition_seed=101),
            "B": RegimeConfig("B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
            "C": RegimeConfig("C", target_density=0.10, persistence=0.25, within_comm_multiplier=4.0, partition_seed=303),
        }
        self.generator = DynamicSBMGenerator(
            num_nodes=90,
            num_communities=3,
            mode="temporal_plus_structure",
            regime_configs=self.regime_configs
        )
        self.calibrator = RegimeCalibrator(
            generator=self.generator,
            max_density_diff=0.05,
            num_samples=100,
            burn_in=30
        )

    def test_transition_matrix_validity(self):
        for name, regime in self.generator.regimes.items():
            a = regime.a_matrix
            b = regime.b_matrix
            self.assertTrue(np.all(a >= 0.0) and np.all(a <= 1.0), f"Invalid a_matrix in regime {name}")
            self.assertTrue(np.all(b >= 0.0) and np.all(b <= 1.0), f"Invalid b_matrix in regime {name}")

    def test_calibration_passes(self):
        report = self.calibrator.calibrate(seed=42)
        self.assertTrue(report.passed, f"Calibration failed: diff = {report.density_diff_AB}")
        self.assertLessEqual(report.density_diff_AB, 0.05)
        self.assertLessEqual(report.density_diff_AC, 0.05)
        self.assertIn("A_vs_B", report.pairwise_similarities)
        self.assertIn("A_vs_C", report.pairwise_similarities)


if __name__ == "__main__":
    unittest.main()
