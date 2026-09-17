"""Unit tests for Canonical Temporal Graph Network (TGN)."""
import unittest
import torch
import numpy as np

from src.models.tgn import TGN
from src.generator.dsbm import DynamicSBMGenerator
from src.evaluation.event_converter import extract_events_from_sequence


class TestTGN(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.num_nodes = 50
        self.model = TGN(
            num_nodes=self.num_nodes,
            node_dim=32,
            memory_dim=32,
            time_dim=32,
            message_dim=32,
            device=self.device
        )

    def test_forward_pass_shape(self):
        src = torch.tensor([0, 1, 2], dtype=torch.long, device=self.device)
        dst = torch.tensor([3, 4, 5], dtype=torch.long, device=self.device)
        ts = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)

        probs = self.model.predict_link_probabilities(src, dst, ts)
        self.assertEqual(probs.shape, (3,))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_memory_update_and_reset(self):
        src = torch.tensor([0, 1], dtype=torch.long, device=self.device)
        dst = torch.tensor([2, 3], dtype=torch.long, device=self.device)
        ts = torch.tensor([1.0, 1.0], dtype=torch.float32, device=self.device)

        init_mem = self.model.memory_bank.get_memory(src).clone()
        self.assertTrue(torch.all(init_mem == 0.0))

        # Perform memory update
        self.model.update_node_memories(src, dst, ts)
        updated_mem = self.model.memory_bank.get_memory(src)
        self.assertFalse(torch.all(updated_mem == 0.0))

        # Reset memory
        self.model.reset_memory()
        reset_mem = self.model.memory_bank.get_memory(src)
        self.assertTrue(torch.all(reset_mem == 0.0))

    def test_event_extraction_chronology(self):
        gen = DynamicSBMGenerator(num_nodes=30, num_communities=3)
        seq = gen.generate([("A", 10), ("B", 10)], seed=42)
        batches = extract_events_from_sequence(seq, seed=42)

        self.assertEqual(len(batches), 20)
        for t, batch in enumerate(batches):
            if len(batch.timestamps) > 0:
                self.assertTrue(np.all(batch.timestamps == float(t)))

    def test_reproducibility(self):
        torch.manual_seed(42)
        m1 = TGN(num_nodes=30, node_dim=16, memory_dim=16, device=self.device)
        src = torch.tensor([0, 1], dtype=torch.long)
        dst = torch.tensor([2, 3], dtype=torch.long)
        ts = torch.tensor([1.0, 1.0])
        p1 = m1.predict_link_probabilities(src, dst, ts)

        torch.manual_seed(42)
        m2 = TGN(num_nodes=30, node_dim=16, memory_dim=16, device=self.device)
        p2 = m2.predict_link_probabilities(src, dst, ts)

        torch.testing.assert_close(p1, p2)


if __name__ == "__main__":
    unittest.main()
