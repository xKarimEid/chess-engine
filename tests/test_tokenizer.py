"""
Testing Basic Tokenizer module
"""


from src.tokenizer.basic_tokenizer import BasicTokenizer

tokenizer = BasicTokenizer()

fen = 'r2k3r/1p2bp2/pN3n2/1B4pp/3p4/1P1P4/P1b1QPPP/R1B1R1K1 b'


def test_encoding():
    """
    Test encoding
    """

    encoded = tokenizer.encode(fen)
    decoded = tokenizer.decode(encoded)

    assert decoded == fen
