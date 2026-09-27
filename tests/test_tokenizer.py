from tokenizer import Tokenizer


def test_tokenize_basic_sentence():
    tok = Tokenizer()
    tokens = tok.tokenize("Hello world.")
    assert tokens == ["Hello", "world", "."]


def test_tokenize_multiple_spaces_and_newlines():
    tok = Tokenizer()
    tokens = tok.tokenize("  Once   upon\n\na time...  ")
    assert tokens == ["Once", "upon", "a", "time", ".", ".", "."]


def test_tokenize_isolated_punctuation():
    tok = Tokenizer()
    tokens = tok.tokenize("Wait! What? No, really.")
    assert tokens == ["Wait", "!", "What", "?", "No", ",", "really", "."]


def test_tokenize_empty_string():
    tok = Tokenizer()
    assert tok.tokenize("") == []
    assert tok.tokenize("   \n\t  ") == []


def test_tokenize_smart_quotes():
    tok = Tokenizer()
    tokens = tok.tokenize("“Yes!” he said.")
    assert tokens == ["“", "Yes", "!", "”", "he", "said", "."]


def test_tokenize_ending_with_word_without_punctuation():
    tok = Tokenizer()
    tokens = tok.tokenize("Hello world")
    assert tokens == ["Hello", "world"]

