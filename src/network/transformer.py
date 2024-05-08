"""
Add docstring
"""

import torch
from torch import nn
import torch.nn.functional as F

# Global params
N_EMBED = 32
HEAD_SIZE = 8
N_HEADS = 4
N_CHARS = 25
N_BINS = 64
N_BLOCKS = 4
DROPOUT = 0.05
CONTEXT_SIZE = 68

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionHead(nn.Module):
    """
    Basic implementation of self attention
    """

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(N_EMBED, HEAD_SIZE, bias = False)
        self.k = nn.Linear(N_EMBED, HEAD_SIZE, bias = False)
        self.v = nn.Linear(N_EMBED, HEAD_SIZE, bias = False)
        #self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Forward function for self attention
        """

        B, T, C = x.shape
        queries = self.q(x) # (B, T, HEAD_SIZE)
        keys = self.k(x) # (B, T, HEAD_SIZE)
        values = self.v(x) # (B, T, HEAD_SIZE)

        # The communication between the nodes
        wei = keys @ queries.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        #wei = self.dropout(wei)

        out = wei @ values # (B, T, HEAD_SIZE)
        return out

class MultiHeadAttention(nn.Module):
    """
    Basic implementation of multihead attention
    """

    def __init__(self):
        super().__init__()
        assert N_EMBED == N_HEADS * HEAD_SIZE
        self.sa_heads = nn.ModuleList([AttentionHead() for _ in range(N_HEADS)])
        self.proj = nn.Linear(N_EMBED, N_EMBED, bias = True)
        #self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Forward x through all the attention heads and concatenate the results along the
        channel dimension
        """
        x = torch.cat([h(x) for h in self.sa_heads], dim=-1)
        x = self.proj(x)
        return x

class FFWD(nn.Module):
    """
    A basic feed forward module to use after multihead attention
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, N_EMBED*4, bias = True),
            nn.ReLU(),
            nn.Linear(N_EMBED*4, N_EMBED, bias = True),
            #nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        """
        Add docs
        """

        return self.net(x)

class Block(nn.Module):
    """
    A block stitching together layer norms, 
    multihead attention and feedforward modules
    """

    def __init__(self):
        super().__init__()
        self.mh_attention = MultiHeadAttention()
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ffwd = FFWD()
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):

        x = self.mh_attention(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x

class Network(nn.Module):
    """
    Creating the transformer network by initializing the embedding
    matrices, blocks and the head
    """
    def __init__(self):
        super().__init__()
        # Chess piece embedding
        self.embed = nn.Embedding(N_CHARS, N_EMBED)
        # Chess squares embedding
        self.positional_embedding = nn.Embedding(CONTEXT_SIZE, N_EMBED)
        # Self Attention
        self.blocks = nn.Sequential(*[Block() for _ in range(N_BLOCKS)])
        # Flatten out to stretch the rows to all channels for all T
        self.flatten = nn.Flatten(start_dim = 1)
        self.ln = nn.LayerNorm(CONTEXT_SIZE*N_EMBED)
        self.head = nn.Linear(CONTEXT_SIZE*N_EMBED, N_BINS)

    def forward(self, x, targets = None):
        """
        Forwards tokenized FEN notations and outputs the logits
        for each class
        """
        # x is (B, T)
        piece_embed = self.embed(x) # (B, T, N_EMBED)
        positional_embed = self.positional_embedding(torch.arange(CONTEXT_SIZE, device = device))
        # Adding together piece embedding and positional embedding
        x = piece_embed + positional_embed # (B, T, N_EMBED)
        x = self.blocks(x) # (B, T, HEAD_SIZE)
        x = self.flatten(x) # (B, T*HEAD_SIZE)
        x = self.ln(x)
        logits = self.head(x) # (B, N_BINS)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def get_best_position_idx(self, x):
        """
        We need to find the position with the highest evaluation
        from all given positions. To do this we first need to find the 
        binned evaluation for each position and then choose the position with the
        highest binned evaluation
        """

        # Find probs for each position
        logits, loss = self.forward(x) # B, n_bins
        probs = F.softmax(logits, dim = -1)
        # Assign position evaluation based on the highest probability
        binned_eval = torch.argmax(probs, dim = 1)
        # Find the position idx with the best evaluation (highest binned evaluation)
        best_position = torch.argmin(binned_eval)

        return best_position
