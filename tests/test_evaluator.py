import math
import pytest
from evaluator import Evaluator


def test_evaluator_metrics():
    evaluator = Evaluator()

    model = {
        "<START>": {"the": 1.0},
        "the": {"cat": 0.5, "dog": 0.5},
        "cat": {"sat": 1.0},
        "dog": {"barked": 1.0},
        "sat": {".": 1.0},
        "barked": {".": 1.0},
        ".": {"<END>": 1.0},
    }

    assert evaluator.unique_bigram_count(model) == 8
    assert evaluator.vocabulary_size(model) == 8

    # Test token sequence with 1 unseen bigram ("cat" -> "barked")
    test_tokens = ["<START>", "the", "cat", "barked", "."]
    metrics = evaluator.evaluate(model, test_tokens)

    assert metrics["evaluated_bigrams"] == 4
    assert metrics["seen_bigrams"] == 3
    assert metrics["unseen_bigrams"] == 1
    assert metrics["coverage"] == 0.75
    assert metrics["total_log_probability"] == -math.inf
    assert metrics["perplexity"] == math.inf


def test_evaluator_empty_and_single_token():
    evaluator = Evaluator()
    model = {"<START>": {"hello": 1.0}, "hello": {"<END>": 1.0}}

    empty_res = evaluator.evaluate(model, [])
    assert empty_res["evaluated_bigrams"] == 0
    assert empty_res["coverage"] == 0.0

    single_res = evaluator.evaluate(model, ["hello"])
    assert single_res["evaluated_bigrams"] == 0


def test_evaluator_perplexity_overflow_protection():
    evaluator = Evaluator()
    assert evaluator.calculate_perplexity(-100000.0) == math.inf
    assert evaluator.calculate_perplexity(-math.inf) == math.inf
    assert evaluator.calculate_perplexity(0.0) == 1.0


def test_evaluator_validate_probabilities_errors():
    evaluator = Evaluator()

    # Negative probability
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        evaluator.validate_probabilities({"word": {"next": -0.5}})

    # Probability greater than 1
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        evaluator.validate_probabilities({"word": {"next": 1.5}})

    # Sum does not equal 1.0
    with pytest.raises(ValueError, match="expected approximately 1.0"):
        evaluator.validate_probabilities({"word": {"a": 0.2, "b": 0.2}})


def test_evaluator_get_bigrams():
    evaluator = Evaluator()
    assert evaluator.get_bigrams(["a", "b", "c"]) == [("a", "b"), ("b", "c")]
    assert evaluator.get_bigrams(["single"]) == []
    assert evaluator.get_bigrams([]) == []


def test_evaluator_report(capsys):
    evaluator = Evaluator()
    metrics = {
        "vocabulary_size": 100,
        "unique_bigram_count": 500,
        "evaluated_bigrams": 1000,
        "seen_bigrams": 950,
        "unseen_bigrams": 50,
        "coverage": 0.95,
        "total_log_probability": -250.0,
        "average_log_probability": -0.25,
        "perplexity": 1.28,
    }
    evaluator.report(metrics)
    captured = capsys.readouterr()
    assert "===== Bigram Model Evaluation =====" in captured.out
    assert "Coverage: 95.00%" in captured.out
    assert "Perplexity: 1.28" in captured.out


def test_evaluator_probability_lookup():
    evaluator = Evaluator()
    model = {"hello": {"world": 1.0}}
    assert evaluator.get_probability(model, "missing", "word") == 0.0
    assert evaluator.get_probability(model, "hello", "missing") == 0.0
    assert evaluator.get_probability(model, "hello", "world") == 1.0


def test_evaluator_coverage_zero_total():
    evaluator = Evaluator()
    assert evaluator.calculate_coverage(0, 0) == 0.0


def test_evaluator_calculate_log_probabilities():
    evaluator = Evaluator()
    model = {"a": {"b": 0.5}, "b": {"c": 0.5}}
    # Sequence with length < 2
    res_single = evaluator.calculate_log_probabilities(model, ["a"])
    assert res_single["total_log_probability"] == 0.0
    assert res_single["average_log_probability"] == 0.0

    # Sequence with all seen bigrams
    res_seen = evaluator.calculate_log_probabilities(model, ["a", "b", "c"])
    expected_total = math.log(0.5) + math.log(0.5)
    assert math.isclose(res_seen["total_log_probability"], expected_total)
    assert math.isclose(res_seen["average_log_probability"], expected_total / 2)

