import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward_neural_network import FeedForwardNeuralNetwork
from residual_connection import ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ffn = FeedForwardNeuralNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.rc_ffn = ResidualConnection(dropout=dropout, d_model=d_model)
        self.rc_mha = ResidualConnection(dropout=dropout, d_model=d_model)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        x = self.rc_mha(x, lambda x: self.mha(x, x, x, mask))
        x = self.rc_ffn(x, self.ffn)
        return x
