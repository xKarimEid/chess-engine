"""
Simple implementation of a chess engine that returns a random legal move
when given a position in fen notation
"""

from flask import Flask, request
import random

import chess


app = Flask(__name__)

@app.route('/make_random_move', methods = ['GET'])
def make_random_move():
    """
    Takes a fen position and makes a legal random move and outputs
    the position after the move has been made
    """
    data = request.json
    fen = data['fen']
    new_position = get_new_position(fen)

    return new_position

def get_random_move(fen):
    """
    Given a position, this function returns a random legal move
    in uci notation
    """
    board = chess.Board(fen)
    moves = [move for move in board.legal_moves]
    random_move = random.choice(moves)
    return random_move

def get_new_position(fen):
    """
    Returns the new position after a random legal 
    move has been made
    """

    move = get_random_move(fen)
    board = chess.Board(fen)
    board.push(move)
    return board.fen()

