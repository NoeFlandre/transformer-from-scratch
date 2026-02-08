import torch.nn as nn
import torch

class ProjectionLayer(nn.Module):

    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x): # (batch_size, seq_length, d_model) -> (batch_size, seq_length, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
