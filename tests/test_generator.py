from generator import Generator


def test_generator_word_not_in_probabilities():
    generator = Generator()
    probs = {"hello": {"world": 1.0}}
    assert generator.generate(probs, "unknown_word") is None


def test_generator_word_in_probabilities():
    generator = Generator()
    probs = {"hello": {"world": 1.0}}
    assert generator.generate(probs, "hello") == "world"


def test_generator_generate_sentence_normal():
    generator = Generator()
    probs = {
        "<START>": {"Once": 1.0},
        "Once": {"upon": 1.0},
        "upon": {"a": 1.0},
        "a": {"time": 1.0},
        "time": {".": 1.0},
        ".": {"<END>": 1.0},
    }
    sentence = generator.generate_sentence(probs, max_tokens=20)
    assert sentence == "Once upon a time."


def test_generator_generate_sentence_max_tokens_cutoff():
    generator = Generator()
    # Cyclic transition that would loop forever without max_tokens limit
    probs = {
        "<START>": {"loop": 1.0},
        "loop": {"loop": 1.0},
    }
    sentence = generator.generate_sentence(probs, max_tokens=3)
    assert sentence == "loop loop loop"
