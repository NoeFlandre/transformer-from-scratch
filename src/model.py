import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from encoder_block import EncoderBlock
from decoder_block import DecoderBlock
from multi_head_attention import MultiHeadAttention
from feed_forward_neural_network import FeedForwardNeuralNetwork
from projection_layer import ProjectionLayer
from transformer_block import TransformerBlock
from input_embedding import InputEmbedding
from positional_encoding import PositionalEncoding

def build_transformer(
    source_vocab_size: int,
    target_vocab_size: int,
    source_seq_length: int,
    target_seq_length: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048):

    source_embedding = InputEmbedding(d_model, source_vocab_size)
    target_embedding = InputEmbedding(d_model, target_vocab_size)

    source_position = PositionalEncoding(d_model, source_seq_length, dropout)
    target_position = PositionalEncoding(d_model, target_seq_length, dropout)

    decoder_layers = nn.ModuleList([DecoderBlock(self_attention_block=MultiHeadAttention(d_model, h, dropout), cross_attention_block = MultiHeadAttention(d_model, h, dropout), feed_forward_block = FeedForwardNeuralNetwork(d_model, d_ff, dropout), dropout=dropout, d_model=d_model) for _ in range(N)])

    encoder = Encoder(d_model, d_ff, h, N, dropout)
    decoder = Decoder(decoder_layers, d_model)

    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    transformer = TransformerBlock(encoder, decoder, projection_layer, source_embedding, target_embedding, source_position, target_position)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # xavier initialization weights such that the variance of the activations remains roughly the same across layers. This keeps the training stable from the very first epoch
    return transformer
