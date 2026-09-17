"""Node memory bank, message function, and memory updater for TGN."""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .time_encoder import TimeEncoder


class MemoryBank(nn.Module):
    """
    Manages node states s_i(t) and last-updated timestamps t_i.
    """

    def __init__(self, num_nodes: int, memory_dim: int, device: torch.device):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.device = device

        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim, device=device))
        self.register_buffer("last_update", torch.zeros(num_nodes, device=device))

    def reset_memory(self) -> None:
        """Reset all node memories to zeros."""
        self.memory.fill_(0.0)
        self.last_update.fill_(0.0)

    def detach_memory(self) -> None:
        """Detach memory tensors from computation graph."""
        self.memory.detach_()

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.memory[node_ids]

    def set_memory(self, node_ids: torch.Tensor, updated_memory: torch.Tensor) -> None:
        self.memory[node_ids] = updated_memory

    def get_last_update(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.last_update[node_ids]

    def set_last_update(self, node_ids: torch.Tensor, timestamps: torch.Tensor) -> None:
        self.last_update[node_ids] = timestamps


class MessageFunction(nn.Module):
    """
    Computes interaction messages between interacting nodes:
    m_u(t) = MLP([s_u(t^-), s_v(t^-), delta_t, e_uv])
    """

    def __init__(self, memory_dim: int, time_dim: int, edge_feat_dim: int, message_dim: int):
        super().__init__()
        input_dim = 2 * memory_dim + time_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        time_encoding: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([src_memory, dst_memory, time_encoding, edge_features], dim=-1)
        return self.mlp(inputs)


class MemoryUpdater(nn.Module):
    """
    Updates node memory using GRUCell:
    s_u(t) = GRUCell(aggregated_message, s_u(t^-))
    """

    def __init__(self, message_dim: int, memory_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def forward(self, messages: torch.Tensor, previous_memory: torch.Tensor) -> torch.Tensor:
        return self.gru(messages, previous_memory)
