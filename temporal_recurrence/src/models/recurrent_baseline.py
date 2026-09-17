"""Simple Recurrent Baseline (GRU Node Memory) for Temporal Link Prediction."""
from typing import Optional
import torch
import torch.nn as nn


class GRUTemporalBaseline(nn.Module):
    """
    A minimal generic recurrent temporal baseline that updates node representations
    using a GRUCell without graph message passing.
    """

    def __init__(
        self,
        num_nodes: int = 300,
        node_dim: int = 32,
        hidden_dim: int = 32,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.node_embedding = nn.Embedding(num_nodes, node_dim)
        # Recurrent cell updating state from degree/interaction activity summary
        self.gru = nn.GRUCell(input_size=node_dim + 1, hidden_size=hidden_dim)
        self.register_buffer("memory", torch.zeros(num_nodes, hidden_dim, device=device))

        total_repr_dim = 2 * (node_dim + hidden_dim)
        self.link_decoder = nn.Sequential(
            nn.Linear(total_repr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.to(device)

    def reset_memory(self) -> None:
        self.memory.fill_(0.0)

    def detach_memory(self) -> None:
        self.memory.detach_()

    def compute_node_representations(self, node_ids: torch.Tensor, current_timestamps: torch.Tensor) -> torch.Tensor:
        static_embed = self.node_embedding(node_ids)
        mem = self.memory[node_ids]
        return torch.cat([static_embed, mem], dim=-1)

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
        if len(src_nodes) == 0:
            return

        all_nodes = torch.cat([src_nodes, dst_nodes])
        unique_nodes, counts = torch.unique(all_nodes, return_counts=True)
        counts_norm = (counts.float().unsqueeze(-1) / 10.0)  # Normalized activity signal

        node_embeds = self.node_embedding(unique_nodes)
        inputs = torch.cat([node_embeds, counts_norm], dim=-1)
        prev_h = self.memory[unique_nodes]

        updated_h = self.gru(inputs, prev_h)
        self.memory[unique_nodes] = updated_h.detach()
