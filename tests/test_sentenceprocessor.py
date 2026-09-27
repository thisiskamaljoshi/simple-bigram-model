from sentenceprocessor import SentenceProcessor


def test_sentence_processor_single_sentence():
    sp = SentenceProcessor()
    tokens = ["Hello", "world", "."]
    result = sp.process(tokens)
    assert result == ["<START>", "Hello", "world", ".", "<END>"]


def test_sentence_processor_multiple_sentences():
    sp = SentenceProcessor()
    tokens = ["Hello", "world", ".", "How", "are", "you", "?"]
    result = sp.process(tokens)
    assert result == [
        "<START>", "Hello", "world", ".", "<END>",
        "<START>", "How", "are", "you", "?", "<END>",
    ]


def test_sentence_processor_consecutive_punctuation():
    sp = SentenceProcessor()
    tokens = ["Wait", ".", ".", ".", "Go", "!"]
    result = sp.process(tokens)
    assert result == [
        "<START>", "Wait", ".", ".", ".", "<END>",
        "<START>", "Go", "!", "<END>",
    ]


def test_sentence_processor_orphan_leading_punctuation():
    sp = SentenceProcessor()
    tokens = [".", "Hello", "."]
    result = sp.process(tokens)
    assert result == ["<START>", "Hello", ".", "<END>"]


def test_sentence_processor_missing_trailing_punctuation():
    sp = SentenceProcessor()
    tokens = ["Hello", "world"]
    result = sp.process(tokens)
    assert result == ["<START>", "Hello", "world", "<END>"]


def test_sentence_processor_empty():
    sp = SentenceProcessor()
    assert sp.process([]) == []
