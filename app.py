"""
Simple implementation of a chess engine that returns a random legal move
when given a position in fen notation
"""


from flask import Flask, request, jsonify
from flask_cors import CORS

from src.engine import Engine

# Initialize an instance of the app
app = Flask(__name__)
# Allow requests from a specific address
CORS(app, origins = 'http://127.0.0.1:5000')

# Initialize an instance of the chess engine
engine = Engine()

@app.route('/make_move', methods = ['POST'])
def make_move():
    """
    Takes a fen position and makes a legal move and outputs
    the position after the move has been made
    """

    fen = request.json['fen']
    new_fen = engine.get_best_position(fen)

    return jsonify({"fen": new_fen})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
