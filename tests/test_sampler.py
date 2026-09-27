from unittest.mock import patch
from sampler import Sampler


def test_sampler_empty_probabilities():
    sampler = Sampler()
    assert sampler.sample({}) is None


def test_sampler_greedy_low_temperature():
    sampler = Sampler()
    probabilities = {"cat": 0.1, "dog": 0.8, "bird": 0.1}
    # Low temperature (<= 0.05) should deterministically pick the highest probability token
    assert sampler.sample(probabilities, temperature=0.01) == "dog"


def test_sampler_temperature_scaling():
    sampler = Sampler()
    probabilities = {"cat": 0.3, "dog": 0.7}
    # Valid sample returned under custom temperature
    result = sampler.sample(probabilities, temperature=0.5)
    assert result in probabilities


def test_sampler_fallback_on_exception():
    sampler = Sampler()
    probabilities = {"cat": 0.2, "dog": 0.8}
    with patch("random.choices", side_effect=ValueError("Simulated sampling error")):
        # Should fallback to argmax ("dog")
        result = sampler.sample(probabilities, temperature=1.0)
        assert result == "dog"
