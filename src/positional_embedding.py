import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of size (seq_length, d_model) because we want our sentences to be of length seq_length and each positional embedding to be of size d_model
        # position 0 -> [0,0,0...,0]
        # position 1 -> [0,0,0...,0]
        # position 2 -> [0,0,0...,0]
        # ...
        pe = torch.zeros(seq_length, d_model)  # (seq_length, d_model)
        position = torch.arange(
            0, seq_length, dtype=torch.float
        ).unsqueeze(
            1
        )  # torch.arange(0, seq_length) is going to create a tensor of shape (seq_length,) -> [0,1,2,..seq_length-1] however we are going to divide each position by a denominator. Hence we perform unsqueeze(1) which gives us (seq_length, 1) -> [[0],[1],...,[seq_length-1]] so positions are turned into a column vector so PyTorch can generate all (position x frequency) pairs via broadcasting
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # this is just a numerical way (by taking the exponential of the log) of computing the division term : 1/(10000**(2i/d_model)) where i is the frequency
        # Here position * div_term : (seq_length, d_model/2) each row is a position while each column is a frequency
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odds dimensions
        pe = pe.unsqueeze(0)  # add the batch dimension : (1, seq_length, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + (self.pe[:, : x.size(1)]).requires_grad_(
            False
        )  # we don't want to learn this
        return self.dropout(x)
