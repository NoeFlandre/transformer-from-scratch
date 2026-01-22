import torch
import torch.nn as nn
from layer_normalization import LayerNormalization


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float, d_model: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model=d_model)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        x = x + self.dropout(sublayer(self.norm(x)))
        return x
