import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self,  d_model : int, eps : float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # Multiplicative
        self.beta = nn.Parameter(torch.zeros(d_model)) # Additive

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # as a reminder x has dimension (batch_size, seq_length, d_model)
        mean = x.mean(dim=-1, keepdim=True) # we compute the mean but keep the dimension
        std = x.std(dim=-1, keepdim=True, unbiased=False) # by default the Bessel's correction s applied to the std for statistical estimation which is not the case for layer norm
        return self.alpha * (x-mean)/(std + self.eps) + self.beta