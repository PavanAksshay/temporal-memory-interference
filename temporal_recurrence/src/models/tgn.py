"""Canonical Temporal Graph Network (TGN) Model."""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .time_encoder import TimeEncoder
from .memory import MemoryBank, MessageFunction, MemoryUpdater


class TGN(nn.Module):
    """
    Canonical Temporal Graph Network (TGN) architecture with node embeddings,
    continuous-time memory bank, message passing, GRU memory updates, and link decoder.
    """

    def __init__(
        self,
        num_nodes: int = 300,
        node_dim: int = 64,
        memory_dim: int = 64,
        time_dim: int = 64,
        message_dim: int = 64,
        edge_feat_dim: int = 1,
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

        # Link prediction decoder: takes concatenated representations of src and dst
        total_repr_dim = 2 * (node_dim + memory_dim)
        self.link_decoder = nn.Sequential(
            nn.Linear(total_repr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.to(device)

    def reset_memory(self) -> None:
        self.memory_bank.reset_memory()

    def detach_memory(self) -> None:
        self.memory_bank.detach_memory()

    def compute_node_representations(self, node_ids: torch.Tensor, current_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute combined static node embedding and dynamic memory for given nodes:
        h_u(t) = [x_u, s_u(t)]
        """
        static_embed = self.node_embedding(node_ids)
        memories = self.memory_bank.get_memory(node_ids)
        return torch.cat([static_embed, memories], dim=-1)

    def predict_link_probabilities(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link probability in [0, 1].
        """
        h_src = self.compute_node_representations(src_nodes, timestamps)
        h_dst = self.compute_node_representations(dst_nodes, timestamps)
        edge_repr = torch.cat([h_src, h_dst], dim=-1)
        logits = self.link_decoder(edge_repr).squeeze(-1)
        return torch.sigmoid(logits)

    def predict_logits(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Return raw logits for BCEWithLogitsLoss."""
        h_src = self.compute_node_representations(src_nodes, timestamps)
        h_dst = self.compute_node_representations(dst_nodes, timestamps)
        edge_repr = torch.cat([h_src, h_dst], dim=-1)
        return self.link_decoder(edge_repr).squeeze(-1)

    def update_node_memories(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> None:
        """
        Aggregates messages arriving at interacting nodes and updates memory (Executed strictly after prediction).
        """
        if len(src_nodes) == 0:
            return

        if edge_features is None:
            edge_features = torch.ones((len(src_nodes), self.edge_feat_dim), device=self.device)

        h_src = self.compute_node_representations(src_nodes, timestamps)
        h_dst = self.compute_node_representations(dst_nodes, timestamps)

        src_last = self.memory_bank.get_last_update(src_nodes)
        dst_last = self.memory_bank.get_last_update(dst_nodes)

        dt_src = timestamps - src_last
        dt_dst = timestamps - dst_last

        time_enc_src = self.time_encoder(dt_src)
        time_enc_dst = self.time_encoder(dt_dst)

        # Messages arriving at src and dst
        msg_for_src = self.message_function(h_src, h_dst, time_enc_src, edge_features)
        msg_for_dst = self.message_function(h_dst, h_src, time_enc_dst, edge_features)

        all_target_nodes = torch.cat([src_nodes, dst_nodes])
        all_messages = torch.cat([msg_for_src, msg_for_dst], dim=0)

        # Mean aggregation per unique node
        unique_nodes, inverse_indices = torch.unique(all_target_nodes, return_inverse=True)
        num_unique = len(unique_nodes)

        agg_messages = torch.zeros((num_unique, self.message_dim), device=self.device)
        counts = torch.zeros((num_unique, 1), device=self.device)

        agg_messages.index_add_(0, inverse_indices, all_messages)
        counts.index_add_(0, inverse_indices, torch.ones((len(all_target_nodes), 1), device=self.device))
        agg_messages = agg_messages / torch.clamp(counts, min=1.0)

        prev_unique_mem = self.memory_bank.get_memory(unique_nodes)
        updated_unique_mem = self.memory_updater(agg_messages, prev_unique_mem)

        self.memory_bank.set_memory(unique_nodes, updated_unique_mem.detach())
        current_time_val = timestamps[0]
        self.memory_bank.set_last_update(unique_nodes, torch.full((num_unique,), current_time_val, device=self.device))


class TGNNoMemory(nn.Module):
    """
    Ablation variant of TGN where persistent node memory across events is removed.
    Retains identical node embeddings, time encoder, and link decoder architecture.
    """

    def __init__(
        self,
        num_nodes: int = 300,
        node_dim: int = 64,
        time_dim: int = 64,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.time_dim = time_dim
        self.device = device

        self.node_embedding = nn.Embedding(num_nodes, node_dim)
        self.time_encoder = TimeEncoder(time_dim)

        total_repr_dim = 2 * node_dim
        self.link_decoder = nn.Sequential(
            nn.Linear(total_repr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.to(device)

    def reset_memory(self) -> None:
        pass

    def detach_memory(self) -> None:
        pass

    def compute_node_representations(self, node_ids: torch.Tensor, current_timestamps: torch.Tensor) -> torch.Tensor:
        return self.node_embedding(node_ids)

    def predict_link_probabilities(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        h_src = self.compute_node_representations(src_nodes, timestamps)
        h_dst = self.compute_node_representations(dst_nodes, timestamps)
        edge_repr = torch.cat([h_src, h_dst], dim=-1)
        logits = self.link_decoder(edge_repr).squeeze(-1)
        return torch.sigmoid(logits)

    def predict_logits(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        h_src = self.compute_node_representations(src_nodes, timestamps)
        h_dst = self.compute_node_representations(dst_nodes, timestamps)
        edge_repr = torch.cat([h_src, h_dst], dim=-1)
        return self.link_decoder(edge_repr).squeeze(-1)

    def update_node_memories(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> None:
        # No persistent memory across time steps
        pass

