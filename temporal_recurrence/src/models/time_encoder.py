"""Time encoding layer using Fourier / Bochner random features."""
import math
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """
    Encodes continuous time intervals into dense vector representations
    using Fourier / Bochner features: phi(dt) = cos(w * dt + b).
    """

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        # Learnable frequencies and phase offsets
        self.w = nn.Parameter(torch.randn(dimension))
        self.b = nn.Parameter(torch.randn(dimension))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        delta_t: Tensor of shape (B,) or (B, 1) containing time differences.
        Returns: Tensor of shape (B, dimension).
        """
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)
        # Shape: (B, dimension)
        output = torch.cos(delta_t * self.w + self.b)
        return output
