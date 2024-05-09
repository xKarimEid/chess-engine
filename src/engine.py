"""
Create a chess engine by initiating the tokenizer and
loading the neural network with its trained weights.
"""

import os
import torch

import chess

from src.network.transformer import Network
from src.tokenizer.basic_tokenizer import BasicTokenizer

# Path for saved model weights
model_dir = os.path.join(os.path.dirname(__file__), f'trained_model/model_test.pkl')
# Load the model on cpu
checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
# Initiate the model
model = Network()
# Load the weights of the trained model
model.load_state_dict(checkpoint['model'])

class Engine:
    """
    A chess engine which finds the move with the highest
    winning chess in a given position. 

    The chess engine does the following:
        1) Finds all positions one move ahead from current position
        2) Tokenizes all the positions
        3) Forwards them through a trained neural network
        4) Selects the position with the highest winning percentage
    """

    def __init__(self):
        self.model = model
        self.tokenizer = BasicTokenizer()

    @staticmethod
    def get_all_positions_from_current(fen):
        """
        Finds all legal positions from current position by looking
        one move ahead. 
        """

        # Create the current position on the board
        board = chess.Board(fen)
        positions = []
        # Iterate through all legal moves and push
        # them to the board 
        for move in board.legal_moves:
            board.push(move)
            positions.append(board.fen())
            # Pop the move out after saving the new position
            board.pop()

        return positions

    def tokenize_positions(self, positions):
        """
        Tokenizes the positions to feed them into a neural network
        """

        # Tokenize the positions
        encodings = self.tokenizer.encode(positions)
        # Convert to pytorch tensors
        x = torch.tensor(encodings)

        return x

    def get_best_position(self, fen):
        """
        Orchestrates the functions above to find the best position
        """

        # Get a list of all positions one move ahead from 
        # current position
        positions = self.get_all_positions_from_current(fen)
        # Tokenize the list of positions
        encoded_positions = self.tokenize_positions(positions)
        # Convert to int types. This is necessary for emb layers
        encoded_positions = encoded_positions.long()
        # Get the idx of the best position
        best_position_idx = self.model.get_best_position_idx(encoded_positions)

        return positions[best_position_idx]
