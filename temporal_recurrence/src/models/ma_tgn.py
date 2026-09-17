"""End-to-End Learnable Memory-Augmented Temporal Graph Network (MA-TGN).

Directly overcomes Limitation 6 (Diagnostic Probe -> Differentiable Deep Architecture)
and Limitation 3/15 (Recency Bias & Recovery Inertia in Recurrent TGNNs).
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_encoder import TimeEncoder
from .memory import MemoryBank, MessageFunction, MemoryUpdater
from ..evaluation.prediction import verify_no_future_leakage


class DifferentiableEpisodicBank(nn.Module):
    """
    Maintains a differentiable external memory bank of historical graph/node states
    with strict temporal indexing to prevent future information leakage.
    """

    def __init__(self, num_nodes: int, memory_dim: int, max_history_size: int = 50, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.max_history_size = max_history_size
        self.device = device

        self.timestamps: List[int] = []
        # Stored historical node memory tensors: list of (N, memory_dim)
        self.stored_node_memories: List[torch.Tensor] = []
        # Stored historical graph key summaries: list of (key_dim,)
        self.stored_graph_keys: List[torch.Tensor] = []

    def reset(self) -> None:
        self.timestamps.clear()
        self.stored_node_memories.clear()
        self.stored_graph_keys.clear()

    def store_snapshot(self, t: int, node_memories: torch.Tensor, graph_key: torch.Tensor) -> None:
        """Store a historical memory checkpoint."""
        if len(self.timestamps) > 0 and t <= self.timestamps[-1]:
            # Replace if same timestep or ignore
            return
        
        if len(self.timestamps) >= self.max_history_size:
            # FIFO eviction of oldest if capacity exceeded
            self.timestamps.pop(0)
            self.stored_node_memories.pop(0)
            self.stored_graph_keys.pop(0)

        self.timestamps.append(t)
        self.stored_node_memories.append(node_memories.detach().clone())
        self.stored_graph_keys.append(graph_key.detach().clone())

    def get_valid_history(self, current_time: int) -> Tuple[List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns all historical checkpoints strictly prior to current_time.
        Guarantees zero future leakage.
        """
        valid_indices = [i for i, t in enumerate(self.timestamps) if t < current_time]
        if not valid_indices:
            return [], None, None

        valid_times = [self.timestamps[i] for i in valid_indices]
        verify_no_future_leakage(current_time, valid_times)

        # Stack keys: (H, key_dim)
        keys = torch.stack([self.stored_graph_keys[i] for i in valid_indices], dim=0)
        # Stack node memories: (H, N, memory_dim)
        memories = torch.stack([self.stored_node_memories[i] for i in valid_indices], dim=0)

        return valid_times, keys, memories


class MultiHeadAddressableRetrieval(nn.Module):
    """
    Multi-head differentiable addressable retrieval over historical episodic memory.
    """

    def __init__(self, key_dim: int, query_dim: int, value_dim: int, num_heads: int = 4):
        super().__init__()
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = key_dim // num_heads

        self.W_q = nn.Linear(query_dim, key_dim)
        self.W_k = nn.Linear(key_dim, key_dim)
        self.W_v = nn.Linear(value_dim, value_dim)
        self.out_proj = nn.Linear(value_dim, value_dim)

    def forward(
        self,
        query: torch.Tensor,               # (B, query_dim)
        historical_keys: torch.Tensor,     # (H, key_dim)
        historical_values: torch.Tensor,   # (H, B, value_dim)
        time_lags: torch.Tensor            # (H,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            retrieved_values: (B, value_dim)
            attention_weights: (B, H)
        """
        B = query.size(0)
        H = historical_keys.size(0)

        # Q: (B, key_dim)
        Q = self.W_q(query)
        # K: (H, key_dim)
        K = self.W_k(historical_keys)

        # Compute dot-product attention scores: (B, H)
        # scores[b, h] = (Q[b] . K[h]) / sqrt(head_dim)
        scale = np.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(0, 1)) / scale

        # Softmax over historical checkpoints H
        attn_weights = F.softmax(scores, dim=-1)  # (B, H)

        # Compute weighted sum of values:
        # historical_values: (H, B, value_dim) -> transpose to (B, H, value_dim)
        V = historical_values.transpose(0, 1)  # (B, H, value_dim)
        V = self.W_v(V)

        # Retrieved: (B, value_dim) = sum_h (attn_weights[b, h] * V[b, h, :])
        retrieved = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)  # (B, value_dim)
        retrieved = self.out_proj(retrieved)

        return retrieved, attn_weights


class AdaptiveTemporalGating(nn.Module):
    """
    Learned gating unit that dynamically weights recent recurrent memory s_u(t)
    against retrieved historical memory r_u(t).
    """

    def __init__(self, memory_dim: int, static_dim: int):
        super().__init__()
        gate_input_dim = 2 * memory_dim + static_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        current_memory: torch.Tensor,    # (B, memory_dim)
        retrieved_memory: torch.Tensor,  # (B, memory_dim)
        static_embed: torch.Tensor       # (B, static_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_representation: (B, memory_dim)
            gate_weight: (B, 1) in [0, 1] (1 = pure recent, 0 = pure historical)
        """
        gate_inputs = torch.cat([current_memory, retrieved_memory, static_embed], dim=-1)
        gate = self.gate_mlp(gate_inputs)  # (B, 1)

        fused = gate * current_memory + (1.0 - gate) * retrieved_memory
        return fused, gate


class MATGN(nn.Module):
    """
    Memory-Augmented Temporal Graph Network (MA-TGN).
    Combines:
    1. Continuous-time GRU Node Memory (Recency / Short-term)
    2. Differentiable Addressable Episodic Memory Bank (Historical / Long-term)
    3. Multi-Head Attention Retrieval
    4. Adaptive Learned Gating
    5. Link Decoder
    """

    def __init__(
        self,
        num_nodes: int = 300,
        node_dim: int = 64,
        memory_dim: int = 64,
        time_dim: int = 64,
        message_dim: int = 64,
        edge_feat_dim: int = 1,
        max_history_size: int = 50,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.message_dim = message_dim
        self.edge_feat_dim = edge_feat_dim
        self.device = device

        # Core embeddings & continuous memory
        self.node_embedding = nn.Embedding(num_nodes, node_dim)
        self.time_encoder = TimeEncoder(time_dim)
        self.memory_bank = MemoryBank(num_nodes, memory_dim, device=device)
        self.message_function = MessageFunction(
            memory_dim=node_dim + memory_dim,
            time_dim=time_dim,
            edge_feat_dim=edge_feat_dim,
            message_dim=message_dim
        )
        self.memory_updater = MemoryUpdater(message_dim, memory_dim)

        # Graph summary projection for episodic keys
        self.graph_key_encoder = nn.Sequential(
            nn.Linear(memory_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Episodic Memory & Attention Retrieval
        self.episodic_bank = DifferentiableEpisodicBank(num_nodes, memory_dim, max_history_size, device)
        self.retrieval_module = MultiHeadAddressableRetrieval(
            key_dim=64,
            query_dim=node_dim + memory_dim + time_dim,
            value_dim=memory_dim,
            num_heads=4
        )

        # Adaptive Gating
        self.adaptive_gating = AdaptiveTemporalGating(memory_dim=memory_dim, static_dim=node_dim)

        # Link Prediction Decoder
        total_repr_dim = 2 * (node_dim + memory_dim)
        self.link_decoder = nn.Sequential(
            nn.Linear(total_repr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.to(device)

    def reset_memory(self) -> None:
        self.memory_bank.reset_memory()
        self.episodic_bank.reset()

    def detach_memory(self) -> None:
        self.memory_bank.detach_memory()

    def checkpoint_current_state(self, t: int) -> None:
        """Saves current node memory states into episodic bank."""
        all_node_ids = torch.arange(self.num_nodes, device=self.device)
        all_memories = self.memory_bank.get_memory(all_node_ids)  # (N, memory_dim)
        # Compute global graph key representation (mean pooling through MLP)
        mean_mem = torch.mean(all_memories, dim=0)  # (memory_dim,)
        graph_key = self.graph_key_encoder(mean_mem)  # (64,)
        self.episodic_bank.store_snapshot(t, all_memories, graph_key)

    def compute_node_representations(
        self,
        node_ids: torch.Tensor,
        current_time: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Computes hybrid representation h_u(t) combining static embedding,
        recent GRU memory, and retrieved historical memory via adaptive gating.
        """
        B = node_ids.size(0)
        static_embed = self.node_embedding(node_ids)  # (B, node_dim)
        current_memory = self.memory_bank.get_memory(node_ids)  # (B, memory_dim)

        # Check episodic memory for valid historical checkpoints
        valid_times, hist_keys, hist_node_memories = self.episodic_bank.get_valid_history(current_time)

        if hist_keys is None or len(valid_times) == 0:
            # Cold-start / No history available yet: fallback to pure current memory
            fused_memory = current_memory
            attn_weights = None
            gate = torch.ones((B, 1), device=self.device)
        else:
            # Query encoding: [static_embed, current_memory, time_encoding(current_time)]
            time_enc = self.time_encoder(torch.full((B,), float(current_time), device=self.device))
            query = torch.cat([static_embed, current_memory, time_enc], dim=-1)  # (B, query_dim)

            # Historical values for specific query nodes: (H, B, memory_dim)
            hist_values = hist_node_memories[:, node_ids, :]  # (H, B, memory_dim)
            time_lags = torch.tensor([current_time - t for t in valid_times], device=self.device, dtype=torch.float32)

            retrieved_memory, attn_weights = self.retrieval_module(
                query=query,
                historical_keys=hist_keys,
                historical_values=hist_values,
                time_lags=time_lags
            )

            # Adaptive gating
            fused_memory, gate = self.adaptive_gating(
                current_memory=current_memory,
                retrieved_memory=retrieved_memory,
                static_embed=static_embed
            )

        h_u = torch.cat([static_embed, fused_memory], dim=-1)
        return h_u, attn_weights, gate

    def predict_logits(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        current_time: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Computes raw link prediction logits."""
        h_src, attn_src, gate_src = self.compute_node_representations(src_nodes, current_time)
        h_dst, attn_dst, gate_dst = self.compute_node_representations(dst_nodes, current_time)

        edge_repr = torch.cat([h_src, h_dst], dim=-1)
        logits = self.link_decoder(edge_repr).squeeze(-1)
        return logits, attn_src, gate_src

    def predict_link_probabilities(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        current_time: int
    ) -> torch.Tensor:
        logits, _, _ = self.predict_logits(src_nodes, dst_nodes, current_time)
        return torch.sigmoid(logits)

    def update_node_memories(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> None:
        """Continuous GRU node memory update based on observed interaction events."""
        if edge_features is None:
            edge_features = torch.ones((len(src_nodes), self.edge_feat_dim), device=self.device)

        unique_nodes = torch.unique(torch.cat([src_nodes, dst_nodes]))
        current_memories = self.memory_bank.get_memory(unique_nodes)
        last_updates = self.memory_bank.get_last_update(unique_nodes)

        # Source -> Destination messages
        src_mem = self.memory_bank.get_memory(src_nodes)
        dst_mem = self.memory_bank.get_memory(dst_nodes)
        src_static = self.node_embedding(src_nodes)
        dst_static = self.node_embedding(dst_nodes)
        full_src = torch.cat([src_static, src_mem], dim=-1)
        full_dst = torch.cat([dst_static, dst_mem], dim=-1)

        delta_t_src = timestamps - self.memory_bank.get_last_update(src_nodes)
        delta_t_dst = timestamps - self.memory_bank.get_last_update(dst_nodes)
        t_enc_src = self.time_encoder(delta_t_src)
        t_enc_dst = self.time_encoder(delta_t_dst)

        msg_src = self.message_function(full_src, full_dst, t_enc_src, edge_features)
        msg_dst = self.message_function(full_dst, full_src, t_enc_dst, edge_features)

        # Aggregate messages
        node_msg_dict: Dict[int, List[torch.Tensor]] = {int(n.item()): [] for n in unique_nodes}
        for u, m in zip(src_nodes, msg_src):
            node_msg_dict[int(u.item())].append(m)
        for v, m in zip(dst_nodes, msg_dst):
            node_msg_dict[int(v.item())].append(m)

        aggregated_msgs = []
        for n in unique_nodes:
            n_id = int(n.item())
            if len(node_msg_dict[n_id]) > 0:
                agg = torch.mean(torch.stack(node_msg_dict[n_id], dim=0), dim=0)
            else:
                agg = torch.zeros(self.message_dim, device=self.device)
            aggregated_msgs.append(agg)

        agg_tensor = torch.stack(aggregated_msgs, dim=0)

        # GRU update
        new_memories = self.memory_updater(agg_tensor, current_memories)
        self.memory_bank.set_memory(unique_nodes, new_memories)
        self.memory_bank.set_last_update(unique_nodes, timestamps[0] if len(timestamps) > 0 else torch.tensor(0.0))
