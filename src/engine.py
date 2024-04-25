"""
Chess engine receives a fen
From this fen all possible fens are generated
The fens are encoded
The fens are forwarded through a model
The model outputs the binned eval for each fen
The chess engine sorts the fens based on the binned eval
Returns the fen with the best eval

"""


import chess
from network.basic_ffwd import FeedForward
from tokenizer.basic_tokenizer import BasicTokenizer

class Engine:
    """
    Chess engine implementation
    """

    def __init__(self):
        self.model = FeedForward()
        self.tokenizer = BasicTokenizer()

    @staticmethod
    def find_all_positions(fen):
        """
        """
        board = chess.Board(fen)
        legal_moves = 

    def find_all_evaluations(self, fens):
        pass

    def find_best_position(self, fen):
        pass

Engine()

