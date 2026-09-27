from bigram import BigramModel
from trainer import Trainer
from sampler import Sampler
from generator import Generator


def test_trainer_and_bigram_model_mle():
    trainer = Trainer()
    bigram_model = BigramModel()

    tokens = ["<START>", "the", "cat", "sat", "on", "the", "mat", "<END>"]
    counts = trainer.train(tokens)
    probs = bigram_model.train(counts)

    # "the" is followed by "cat" once and "mat" once -> 0.5 each
    assert probs["the"]["cat"] == 0.5
    assert probs["the"]["mat"] == 0.5

    # Sum of probabilities for any word must equal 1.0
    for word, transitions in probs.items():
        assert abs(sum(transitions.values()) - 1.0) < 1e-6


def test_bigram_model_empty_and_zero_counts():
    bigram_model = BigramModel()
    assert bigram_model.train({}) == {}
    assert bigram_model.train({"word": {}}) == {}
    assert bigram_model.train({"word": {"a": 0, "b": 0}}) == {}


def test_sampler_greedy_and_temperature():
    sampler = Sampler()
    probs = {"common": 0.9, "rare": 0.1}

    # Greedy argmax
    assert sampler.sample(probs, temperature=0.01) == "common"
    assert sampler.sample({}) is None


def test_generator_deterministic_generation():
    dummy_model = {
        "<START>": {"hello": 1.0},
        "hello": {"world": 1.0},
        "world": {".": 1.0},
        ".": {"<END>": 1.0},
    }
    generator = Generator()
    sentence = generator.generate_sentence(dummy_model, temperature=0.01)
    assert sentence == "hello world."
