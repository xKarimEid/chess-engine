This is a no search based chess-engine which takes a position and 
outputs the best move. The network has not yet been trained here. The engine now
spits out random moves and does not know anything about chess yet. In the next repo
I will add a train/test dataset and train the actual model

# Purpose

How to use this repo?

The neural network is untrained. There will be another repo to train the model and save it

repo 1: Train tokenizer and network -> output a pckle file
repo 2: Build a chess Engine with the tokenizer and pckle file (tokenizer and Network agnostic) - upload to docker?
repo 3: chess engine UI - upload to docker?

repo 4: Putting it all together

# Chess Engine

No search based. Looks at a position and finds the best move by evaluating all positions one move ahead.
Follows Tals principle where he says he only looks one move ahead?

Chess engine is made up of three components

python-chess library for finding all legal moves in a given position.
All legal moves result in different FEN positions.

Tokenizer: Tokenizer takes in a list of FEN notations and encodes them to integers

Neural network: Assigns each position to a specific category. There are 64 different categories where the first
category describes a completely winning position and the last category is for completely loosing positions.

# Tokenizer

Introduce FEN notation

The FEN notation has a maximum length of 71 but can vary in length. To tackle this, each FEN position is padded
to reach the same length. This is a basic tokenizer and will in the next repo use a more advanced tokenizer. 
The BasicTokenzier disregards castling information and the move numbers.

# Neural Net

Binned output


# Libraries

What am I using pthon chess for? 
python chess

# To do 

This repo should be model and tokenizer agnostic
So that one can train a tokenizer and network and just import it to here
