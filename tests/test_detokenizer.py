from detokenizer import Detokenizer


def test_detokenize_no_leading_space():
    detok = Detokenizer()
    text = detok.process(["He", "took", "the", "ball", "."])
    assert text == "He took the ball."
    assert not text.startswith(" ")


def test_detokenize_punctuation_attachment():
    detok = Detokenizer()
    text = detok.process(["Wait", ",", "what", "?", "Really", "!"])
    assert text == "Wait, what? Really!"


def test_detokenize_parentheses_and_brackets():
    detok = Detokenizer()
    assert detok.process(["(", "hello", ")"]) == "(hello)"
    assert detok.process(["[", "test", "]"]) == "[test]"


def test_detokenize_contractions():
    detok = Detokenizer()
    assert detok.process(["It", "'", "s", "great", "!"]) == "It's great!"


def test_detokenize_smart_quotes():
    detok = Detokenizer()
    text = detok.process(["“", "Hello", "!", "”"])
    assert text == "“Hello!”"


def test_detokenize_opening_quote_mid_sentence():
    detok = Detokenizer()
    text = detok.process(["He", "said", "“", "Hello", "”"])
    assert text == "He said “Hello”"


def test_detokenize_empty():
    detok = Detokenizer()
    assert detok.process([]) == ""

