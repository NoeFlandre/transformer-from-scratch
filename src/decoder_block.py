import torch.nn as nn
import torch
from feed_forward_neural_network import FeedForwardNeuralNetwork
from residual_connection import ResidualConnection
from multi_head_attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttention, cross_attention_block : MultiHeadAttention, feed_forward_block : FeedForwardNeuralNetwork, dropout : float, d_model: int):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout, d_model) for _ in range(3)])

    def forward(self, x, encoder_output, source_mask, target_mask): # the source mask (source language) is the mask applied to the encoder while the target mask (target language) is the one applied to the decoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, source_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
        