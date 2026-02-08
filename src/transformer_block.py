import torch
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder
from projection_layer import ProjectionLayer
from input_embedding import InputEmbedding
from positional_encoding import PositionalEncoding


class TransformerBlock(nn.Module):

    def __init__(self, encoder:Encoder, decoder:Decoder, projection_layer:ProjectionLayer, source_embedding:InputEmbedding, target_embedding : InputEmbedding, source_position : PositionalEncoding, target_position : PositionalEncoding ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_position = source_position
        self.target_position = target_position

    def encode(self, source, source_mask):
        source = self.source_embedding(source)
        source = self.source_position(source)
        source = self.encoder(source, source_mask)
        return source

    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_position(target)
        target = self.decoder(target, encoder_output, source_mask, target_mask )
        return target

    def project(self, x):
        return self.projection_layer(x)