import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = (
            d_model  # this is the dimension of each vector associate with each token
        )
        self.vocab_size = (
            vocab_size  # this is the number of tokens we have in our vocabulary
        )
        self.register_buffer(
            "scale", torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        )  # here we are registering the scale variable as a buffer because it is involved in the forward pass but it is not learned, hence registering it as a buffer is helpful as it will be moved to the same device and dtype as the model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )  # a lookup table storing for each token its corresponding vector (num_embedding*embed_dim so here vocab_size*d_model). These embeddings are trainable and learned by the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.embedding(x) * self.scale
        )  # in the original transformer they multiply the embeddings by the square root of the embedding dimension
