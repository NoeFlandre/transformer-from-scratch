
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
from src.model import build_transformer

def test():
    source_vocab_size = 100
    target_vocab_size = 100
    source_seq_length = 10
    target_seq_length = 10
    d_model = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_transformer(source_vocab_size, target_vocab_size, source_seq_length, target_seq_length, d_model).to(device)

    source = torch.randint(0, source_vocab_size, (2, source_seq_length)).to(device) # (batch_size=2, seq_legth=10)
    target = torch.randint(0, target_vocab_size, (2, target_seq_length)).to(device) # (batch_size=2, seq_length=10)

    source_mask = None
    target_mask = None

    print("Running forward pass")
    out = model.encode(source, source_mask)
    print(f"Encoder output shape : {out.shape}, it should be (2, 10, 512)") 

    decoder_out = model.decode(out, source_mask, target, target_mask)
    print(f"Decoder output shape : {decoder_out.shape}, it should be (2, 10, 512)") 

    final_out = model.project(decoder_out)
    print(f"Final output shape : {final_out.shape}, it should be (2, 10, 100)") 

    print("Transformer architecture verified")

if __name__ == "__main__":
    test()

