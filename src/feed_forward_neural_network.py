import torch
import torch.nn as nn 

class FeedForwardNeuralNetwork(nn.Module):
    
    def __init__(self, d_model : int, d_ff : int, dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features = d_model, out_features = d_ff) # first linear layer (W1 and b1)
        self.linear2 = nn.Linear(in_features=d_ff, out_features = d_model) # second linear layer (W2 and b2)
        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
