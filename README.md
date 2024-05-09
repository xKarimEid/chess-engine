# Purpose

This repo is inspired by capablanca when he was reportedly asked how many moves ahead do you calculate, to which he replied: “only one, but it is always the best one”

A tokenizer and a trained neural network can be used together to create a chess engine. 
The chess-engine is a no search-based engine. This means that it doesnt create a monte carlo tree searching many
moves ahead to decide which move is the best move. The way this chess-engine works is by simply looking at all
the positions one move ahead and choosing the move with the highest winning chances based on an evaluation from a trained neural network. 

# Chess Engine

From a given position, all possible positions are found by making all legal moves in the position. The list of FEN positions is tokenized by the tokenizer. The tokenized fens are then forwarded through the neural network. For each position, the output of the neural network is from 0 to 63. Where 0 is loosing and 63 is winning. The binned percentages are in the perspective of the next person to play.

1) All FEN positions one move ahead are created from the given FEN position

2) Tokenizer takes in a list of FEN notations and encodes them to integers

3) Neural network assigns each position to a specific category. There are 64 different categories where the first
category describes a completely losing position and the last category is for completely winning positions.

# Tokenizer

FEN notation intro

The FEN notation has a maximum length of 71 but can vary in length. To tackle this, each FEN position is padded
to reach the same length. This is a basic tokenizer and will in the next repo use a more advanced tokenizer. 
The BasicTokenzier disregards castling information and the move numbers.

# Neural Net

The neural network is tasked with categorizing the position into 64 different classes. Each class representing how winning a position is. The perspective is for the next player to play. So, if it is white to play and the position is completely winning the output would be 63. The output would be the same if it was black to play and the position is completely winning. 