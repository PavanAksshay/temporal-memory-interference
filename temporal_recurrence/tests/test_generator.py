"""Unit tests for temporal graph generator with independent partitions."""
import unittest
import numpy as np

from src.generator.dsbm import DynamicSBMGenerator
from src.generator.regimes import RegimeConfig, compute_regime_affinity_similarity


class TestGenerator(unittest.TestCase):

    def setUp(self):
        self.num_nodes = 60
        self.num_communities = 3
        self.regime_configs = {
            "A": RegimeConfig("A", target_density=0.10, persistence=0.35, within_comm_multiplier=4.0, partition_seed=101),
            "B": RegimeConfig("B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202),
            "C": RegimeConfig("C", target_density=0.10, persistence=0.25, within_comm_multiplier=4.0, partition_seed=303),
        }
        self.generator = DynamicSBMGenerator(
            num_nodes=self.num_nodes,
            num_communities=self.num_communities,
            mode="temporal_plus_structure",
            regime_configs=self.regime_configs
        )

    def test_node_count_and_shape(self):
        seq = self.generator.generate([("A", 10)], seed=42)
        self.assertEqual(seq.snapshots.shape, (10, self.num_nodes, self.num_nodes))

    def test_symmetry(self):
        seq = self.generator.generate([("A", 5), ("B", 5)], seed=42)
        for t in range(seq.total_timesteps):
            G = seq.get_snapshot(t)
            np.testing.assert_array_equal(G, G.T, err_msg=f"Snapshot {t} is not symmetric!")

    def test_no_self_loops(self):
        seq = self.generator.generate([("A", 5), ("B", 5)], seed=42)
        for t in range(seq.total_timesteps):
            G = seq.get_snapshot(t)
            diag = np.diag(G)
            self.assertEqual(np.sum(diag), 0, msg=f"Snapshot {t} contains self-loops!")

    def test_independent_partitions(self):
        regA = self.generator.regimes["A"]
        regB = self.generator.regimes["B"]
        regC = self.generator.regimes["C"]

        self.assertFalse(np.array_equal(regA.community_assignments, regB.community_assignments))
        self.assertFalse(np.array_equal(regA.community_assignments, regC.community_assignments))
        self.assertFalse(np.array_equal(regB.community_assignments, regC.community_assignments))

        # Check that pairwise affinity similarity is lower than 0.70
        sim_AB = compute_regime_affinity_similarity(regA, regB)
        sim_AC = compute_regime_affinity_similarity(regA, regC)
        self.assertLess(sim_AB, 0.70)
        self.assertLess(sim_AC, 0.70)

    def test_reproducibility(self):
        seq1 = self.generator.generate([("A", 15)], seed=123)
        seq2 = self.generator.generate([("A", 15)], seed=123)
        np.testing.assert_array_equal(seq1.snapshots, seq2.snapshots)

    def test_different_seeds(self):
        seq1 = self.generator.generate([("A", 15)], seed=123)
        seq2 = self.generator.generate([("A", 15)], seed=456)
        self.assertFalse(np.array_equal(seq1.snapshots, seq2.snapshots))


if __name__ == "__main__":
    unittest.main()
