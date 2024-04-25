"""
Chess engine receives a fen
From this fen all possible fens are generated
The fens are encoded
The fens are forwarded through a model
The model outputs the binned eval for each fen
The chess engine sorts the fens based on the binned eval
Returns the fen with the best eval

"""

import torch

import chess

from network.basic_ffwd import FeedForward
from tokenizer.basic_tokenizer import BasicTokenizer

class Engine:
    """
    Describe chess engine
    """

    def __init__(self):
        self.model = FeedForward()
        self.tokenizer = BasicTokenizer()

    @staticmethod
    def get_all_positions_from_current(fen):
        """
        Describe function
        """
        board = chess.Board(fen)
        positions = []

        for move in board.legal_moves:
            board.push(move)
            positions.append(board.fen())
            board.pop()

        return positions

    def tokenize_positions(self, positions):
        """
        Describe function
        """
        encodings = self.tokenizer.encode(positions)
        x = torch.tensor(encodings)
        return x

    def get_best_position_idx(self, fen):
        """
        Describe function
        """
        positions = self.get_all_positions_from_current(fen)
        encoded_positions = self.tokenize_positions(positions)
        best_position_idx = self.model.get_best_position_idx(encoded_positions)

        return positions[best_position_idx]
