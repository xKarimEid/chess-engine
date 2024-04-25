"""
Unit test for basic feed forward model
"""

import torch
from src.network.basic_ffwd import FeedForward
from src.tokenizer.basic_tokenizer import BasicTokenizer

# Initialize model to test
model = FeedForward()
# Initialize Tokenizer
tokenizer = BasicTokenizer()

# single FEN
FEN =  'r2qkb1r/pp1b1ppp/2n2n2/1B1pp3/8/2N2N2/PPPPQPPP/R1B1K2R w KQkq -'
# List of FEN notation
FEN_LIST = ['rnbqkb1r/pp3ppp/5n2/3pp3/2B5/2N2N2/PPPPQPPP/R1B1K2R b KQkq -',
            'r1bqkb1r/pp3ppp/2n2n2/3pp3/2B5/2N2N2/PPPPQPPP/R1B1K2R w KQkq -',
            'r1bqkb1r/pp3ppp/2n2n2/1B1pp3/8/2N2N2/PPPPQPPP/R1B1K2R b KQkq -',
            'r2qkb1r/pp1b1ppp/2n2n2/1B1pp3/8/2N2N2/PPPPQPPP/R1B1K2R w KQkq -',
            'r2qkb1r/pp1b1ppp/2n2n2/1B1pN3/8/2N5/PPPPQPPP/R1B1K2R b KQkq -',
            'r2qkb1r/pp1b1ppp/2n2n2/1B2N3/3p4/2N5/PPPPQPPP/R1B1K2R w KQkq -',
            'r2qkb1r/pp1b1ppp/2N2n2/1B6/3p4/2N5/PPPPQPPP/R1B1K2R b KQkq -',
            'r2qk2r/pp1bbppp/2N2n2/1B6/3p4/2N5/PPPPQPPP/R1B1K2R w KQkq -',
            'r2Nk2r/pp1bbppp/5n2/1B6/3p4/2N5/PPPPQPPP/R1B1K2R b KQkq -',
            'r2k3r/pp1bbppp/5n2/1B6/3p4/2N5/PPPPQPPP/R1B1K2R w KQ -']

N_BINS = 64 # This is a configuration setting

def test_forward_pass():
    """
    Testing the forward pass with a single FEN
    """

    # Encode the FEN
    encoded = tokenizer.encode(FEN)
    # Convert to tensor and add a batch dim
    x = torch.tensor(encoded).view(1, 71)
    # Forward pass
    probs = model(x)

    # Assert shape of output is right
    assert probs.shape == (1, N_BINS)

def test_batch_forward_pass():
    """
    Test the forward pass with a batch of encoded FENs
    """

    # Encode the FEN_LIST
    encoded = tokenizer.encode(FEN_LIST)
    # Convert to tensor
    x = torch.tensor(encoded)
    # Forward pass
    probs = model(x)

    # Assert shape of output matches
    assert probs.shape == (len(FEN_LIST), N_BINS)

def test_best_position_idx():
    """
    Test the function of finding the best position 
    from a list of FENs
    """

    encoded = tokenizer.encode(FEN_LIST)
    # Convert to tensor
    x = torch.tensor(encoded)
    # Find the most promising position
    best_idx = model.get_best_position_idx(x)
    print(best_idx)

    assert isinstance(best_idx.item(), int)
