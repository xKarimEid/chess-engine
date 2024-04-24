"""
Tokenizer has the set of string to int
padds the output to a specific length

"""


# All possible characters in a fen adding, '.' for padding purposes
STOI = '. -/12345678BKNPQRbknpqrw'

class BasicTokenizer:
    """
    Basic tokenizer for encoding/decoding a fen position
    """

    def __init__(self):
        self.stoi = {char: idx for char, idx in enumerate(STOI)}
        self.itos = {idx: char for char, idx in enumerate(STOI)}

    def encode(self, fen):
        """
        Encodes a fen to fixed length ints of size 71
        """
        # A fen can have a maximum length of 71
        padding_length = 71 - len(fen)
        # Padding a fen to fixed length with 0
        padded = [0 for _ in range(padding_length)]
        # Converting chars to ints
        ids = [self.stoi[char] for char in fen]
        # Adding the encoded ints to the paddings
        padded.extend(ids)

        return padded

    def decode(self, ids):
        """
        Takes in encoded fen and returns the decoded fen
        """
        # Removing 0 ints as they are paddings
        chars = [self.itos[idx] for idx in ids if idx != 0]
        # Joining chars together to form the fen
        fen = ''.join(chars)

        return fen
