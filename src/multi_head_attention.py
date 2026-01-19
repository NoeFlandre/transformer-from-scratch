import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model : int, num_heads : int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"
        self.dim_head = d_model // num_heads
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model)

    @staticmethod # this means that we can call this method without having any instance of MutiHeadAttention created
    def attention(query, key, value, mask, dropout : nn.Dropout):
        dim_head = query.shape[-1]

        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(dim_head) # (batch_size, num_heads, seq_length, seq_length)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9) # we are replacing all the zero values of the mask with - infinity
        scores = scores.softmax(dim=-1) # we are applying softmax 
        if dropout is not None:
            scores = dropout(scores)

        return scores @ value, scores # we are returning a tuple because we want to also be able to visualize the scores. Here scores @ value is of dimension (batch_size, num_heads, seq_length, dim_head)

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch_size, seq_length, d_model) -> (batch_size, seq_length, d_model)
        key = self.w_k(k) # (batch_size, seq_length, d_model) -> (batch_size, seq_length, d_model)
        value = self.w_v(v) # (batch_size, seq_length, d_model) -> (batch_size, seq_length, d_model)

        # Then we want to split the key, query and value into the different heads

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.dim_head) # (batch_size, seq_length, dmodel) --> (batch_size, seq_length, num_heads, dim_head)
        query = query.transpose(1,2) # (batch_size, seq_length, num_heads, dim_head) --> (batch_size, num_heads, seq_length, dim_head)

        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.dim_head) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, num_heads, dim_head)
        key = key.transpose(1,2) # (batch_size, seq_length, num_heads, dim_head) --> (batch_size, num_heads, seq_length, dim_head)

        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.dim_head) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, num_heads, dim_head)
        value = value.transpose(1,2) # (batch_size, seq_length, num_heads, dim_head) --> (batch_size, num_heads, seq_length, dim_head)

        x, self.scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2) #(batch_size, num_heads, seq_length, dim_head) -> (batch_size, seq_length, num_heads, dim_head)
        x = x.contiguous().view(x.shape[0], x.shape[1], self.num_heads * self.dim_head) # (batch_size, seq_length, d_model)

        return self.w_o(x) # (batch_size, seq_length, d_model) -> (batch_size, seq_length, d_model)