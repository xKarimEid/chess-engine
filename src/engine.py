"""
Chess engine receives a fen
From this fen all possible fens are generated
The fens are encoded
The fens are forwarded through a model
The model outputs the binned eval for each fen
The chess engine sorts the fens based on the binned eval
Returns the fen with the best eval

"""

import os 
import torch

import chess

from src.network.transformer import Network
from src.tokenizer.basic_tokenizer import BasicTokenizer

model_dir = os.path.join(os.path.dirname(__file__), f'trained_model/model_test.pkl')

checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))

model = Network()
model.load_state_dict(checkpoint['model'])

class Engine:
    """
    Describe chess engine
    """

    def __init__(self):
        self.model = model
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

    def get_best_position(self, fen):
        """
        Describe function
        """
        positions = self.get_all_positions_from_current(fen)
        encoded_positions = self.tokenize_positions(positions)
        print(type(encoded_positions),encoded_positions.shape)
        encoded_positions = encoded_positions.long()

        best_position_idx = self.model.get_best_position_idx(encoded_positions)

        return positions[best_position_idx]
