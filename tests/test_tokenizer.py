"""
Unit tests for BasicTokenizer module
"""


from src.tokenizer.basic_tokenizer import BasicTokenizer

# Initialize the BasicTokenizer
tokenizer = BasicTokenizer()

# Example FEN notation for testing
fen = 'r2k3r/1p2bp2/pN3n2/1B4pp/3p4/1P1P4/P1b1QPPP/R1B1R1K1 b'

# List of FEN notations for testing multiple encodings
fen_list = ['rnbqkb1r/pp3ppp/5n2/3pp3/2B5/2N2N2/PPPPQPPP/R1B1K2R b KQkq -',
            'r1bqkb1r/pp3ppp/2n2n2/3pp3/2B5/2N2N2/PPPPQPPP/R1B1K2R w KQkq -',
            'r1bqkb1r/pp3ppp/2n2n2/1B1pp3/8/2N2N2/PPPPQPPP/R1B1K2R b KQkq -',
            'r2qkb1r/pp1b1ppp/2n2n2/1B1pp3/8/2N2N2/PPPPQPPP/R1B1K2R w KQkq -',
            'r2qkb1r/pp1b1ppp/2n2n2/1B1pN3/8/2N5/PPPPQPPP/R1B1K2R b KQkq -',
            'r2qkb1r/pp1b1ppp/2n2n2/1B2N3/3p4/2N5/PPPPQPPP/R1B1K2R w KQkq -',
            'r2qkb1r/pp1b1ppp/2N2n2/1B6/3p4/2N5/PPPPQPPP/R1B1K2R b KQkq -',
            'r2qk2r/pp1bbppp/2N2n2/1B6/3p4/2N5/PPPPQPPP/R1B1K2R w KQkq -',
            'r2Nk2r/pp1bbppp/5n2/1B6/3p4/2N5/PPPPQPPP/R1B1K2R b KQkq -',
            'r2k3r/pp1bbppp/5n2/1B6/3p4/2N5/PPPPQPPP/R1B1K2R w KQ -']

def test_single_encoding():
    """
    Test encoding a single FEN notation
    """

    encoded = tokenizer.encode(fen)
    decoded = tokenizer.decode(encoded)

    assert decoded == fen

def test_multiple_encoding():
    """
    Test encoding multiple FEN notations
    """
    encoded = tokenizer.encode(fen_list)
    decoded = tokenizer.decode(encoded)

    # Check if after encoding/decoding, the same number of FEN
    # strings as the original list
    assert len(decoded) == len(fen_list)

def test_multiple_encoding_2():
    """
    Test encoding multiple fen notations
    """

    # Encode and decode the list of FENs
    encoded = tokenizer.encode(fen_list)
    decoded = tokenizer.decode(encoded)

    extracted_parts = []
    # The tokenization gets rid of castling and move number
    # information. So this information is removed from the
    # original list of FENs to be compared with the decoded FENs
    for fen in fen_list:
        parts = fen.split(" ")
        # Only the first two parts are extracted by the tokenizer
        extracted_part = parts[:2]
        extracted_parts.append(' '.join(extracted_part))

    # Check if after encoding/decoding a list of FENs the original FENs
    # still match the decoded FENs
    assert extracted_parts == decoded
