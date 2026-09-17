"""Unit tests for the Learnable Memory-Augmented TGNN (MA-TGN)."""
import unittest
import torch
import numpy as np

from src.models.ma_tgn import (
    MATGN,
    DifferentiableEpisodicBank,
    MultiHeadAddressableRetrieval,
    AdaptiveTemporalGating
)


class TestMATGN(unittest.TestCase):
    """Test suite for MA-TGN model architecture and causality checks."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.num_nodes = 30
        self.node_dim = 16
        self.memory_dim = 16
        self.time_dim = 16
        self.message_dim = 16

    def test_episodic_bank_causality(self):
        """Verify episodic bank never leaks future timestamps."""
        bank = DifferentiableEpisodicBank(self.num_nodes, self.memory_dim, max_history_size=10, device=self.device)
        
        # Store snapshots at t=5, t=10, t=15
        dummy_mem = torch.randn(self.num_nodes, self.memory_dim)
        dummy_key = torch.randn(64)
        bank.store_snapshot(5, dummy_mem, dummy_key)
        bank.store_snapshot(10, dummy_mem, dummy_key)
        bank.store_snapshot(15, dummy_mem, dummy_key)

        # Query at current_time = 12 -> must only return [5, 10]
        valid_times, keys, memories = bank.get_valid_history(current_time=12)
        self.assertEqual(valid_times, [5, 10])
        self.assertEqual(keys.shape, (2, 64))
        self.assertEqual(memories.shape, (2, self.num_nodes, self.memory_dim))

        # Query at current_time = 4 -> must return empty
        valid_times_0, keys_0, _ = bank.get_valid_history(current_time=4)
        self.assertEqual(valid_times_0, [])
        self.assertIsNone(keys_0)

    def test_multi_head_retrieval(self):
        """Verify multi-head attention retrieval and gradient flow."""
        retrieval = MultiHeadAddressableRetrieval(key_dim=32, query_dim=48, value_dim=16, num_heads=4)
        
        B = 8
        H = 5
        query = torch.randn(B, 48, requires_grad=True)
        keys = torch.randn(H, 32)
        values = torch.randn(H, B, 16)
        time_lags = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])

        out, attn = retrieval(query, keys, values, time_lags)
        self.assertEqual(out.shape, (B, 16))
        self.assertEqual(attn.shape, (B, H))

        # Check softmax sum to 1
        sum_attn = torch.sum(attn, dim=-1)
        self.assertTrue(torch.allclose(sum_attn, torch.ones(B), atol=1e-5))

        # Check gradients backprop
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(query.grad)

    def test_adaptive_gating(self):
        """Verify adaptive gating produces valid [0, 1] interpolation."""
        gating = AdaptiveTemporalGating(memory_dim=16, static_dim=16)
        B = 10
        curr_mem = torch.randn(B, 16)
        ret_mem = torch.randn(B, 16)
        static_emb = torch.randn(B, 16)

        fused, gate = gating(curr_mem, ret_mem, static_emb)
        self.assertEqual(fused.shape, (B, 16))
        self.assertEqual(gate.shape, (B, 1))
        self.assertTrue(torch.all(gate >= 0.0) and torch.all(gate <= 1.0))

    def test_matgn_end_to_end_training(self):
        """Verify full MA-TGN model forward, loss computation, and optimization step."""
        model = MATGN(
            num_nodes=self.num_nodes,
            node_dim=self.node_dim,
            memory_dim=self.memory_dim,
            time_dim=self.time_dim,
            message_dim=self.message_dim,
            device=self.device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 1. Store a historical checkpoint at t=5
        model.checkpoint_current_state(t=5)

        # 2. Simulate interaction batch at t=10
        src = torch.tensor([0, 1, 2], dtype=torch.long)
        dst = torch.tensor([3, 4, 5], dtype=torch.long)
        labels = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

        # Forward pass
        logits, attn, gate = model.predict_logits(src, dst, current_time=10)
        self.assertEqual(logits.shape, (3,))
        self.assertIsNotNone(attn)
        self.assertIsNotNone(gate)

        # Loss and step
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update memories
        model.update_node_memories(src, dst, timestamps=torch.tensor([10.0, 10.0, 10.0]))


if __name__ == "__main__":
    unittest.main()
