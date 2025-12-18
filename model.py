import torch
import torch.nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # this is the dimension of each vector associate with each token
        self.vocab_size = vocab_size # this is the number of tokens we have in our vocabulary
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model) # a lookup table storing for each token its corresponding vector (num_embedding*embed_dim so here vocab_size*d_model). These embeddings are trainable and learned by the model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # in the original transformer they multiply the embeddings by the square root of the embedding dimension