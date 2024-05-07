This is a no search based chess-engine which takes a position and 
predicts its evaluation. 

# Purpose

This repo shows how a tokenizer and a trained neural network can be used together as a chess engine. 

Given a position in FEN notation, it can be tokenized and then fed into a neural network to predict the winning percentage of the given position for the player whos turn it is.

From a given position, all possible positions are found by making all legal moves in the position. The list of FEN positions is tokenized by the tokenizer. The tokenized fens are then forwarded through the neural network. For each position, the output of the neural network is from 0 to 63. Where 0 is loosing and 63 is winning. The binned percentages are in the perspective of the next person to play.

No search based. Looks at a position and finds the best move by evaluating all positions one move ahead.
This is inspired by capablanca when he was asked how many moves ahead do you calculate, to which he replies: “only one, but it is always the best one”

# Chess Engine

python-chess library is used for finding all legal moves in a given position.
All legal moves result in different FEN positions.

Tokenizer: Tokenizer takes in a list of FEN notations and encodes them to integers

Neural network: Assigns each position to a specific category. There are 64 different categories where the first
category describes a completely losing position and the last category is for completely winning positions.

# Tokenizer

FEN notation intro

The FEN notation has a maximum length of 71 but can vary in length. To tackle this, each FEN position is padded
to reach the same length. This is a basic tokenizer and will in the next repo use a more advanced tokenizer. 
The BasicTokenzier disregards castling information and the move numbers.

# Neural Net

The neural network is tasked with categorizing the position into 64 different classes. Each class representing how winning a position is. The perspective is for the next player to play. So, if it is white to play and the position is completely winning the output would be 63. The output would be the same if it was black to play and the position is completely winning. 