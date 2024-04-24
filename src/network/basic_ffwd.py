"""
Basic ffwd model

context length 65?

Takes in a fen ->
fen is int encoded by using a standard encoding method
('p':1 and so forth)
no positional encoding
padded to fixed length

embedding matrix
dense hidden layer with activation
final output layer as binn

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHARS = 21
N_EMBED = 8
N_BINS = 64
CONTEXT_LENGTH = 71

class FeedForward(nn.Module):
    """
    Basic Neural network architecture
    """
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_CHARS, N_EMBED)
        self.flatten = nn.Flatten(end_dim = -1)
        self.hidden = nn.Linear(N_EMBED*CONTEXT_LENGTH, N_BINS)

    def forward(self, x):
        """
        Forward function and get the probs
        """

        x = self.emb(x)
        x = self.flatten(x)
        logits = self.hidden(x)
        probs = F.softmax(logits, dim = -1)

        return probs
