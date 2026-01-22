import torch.nn as nn
from encoder_block import EncoderBlock
from layer_normalization import LayerNormalization

class Encoder(nn.Module):

    def __init__(self, d_model:int, d_ff:int, num_heads:int, num_layers:int, dropout:float):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers: 
            x = layer(x, mask)
        return self.norm(x)
