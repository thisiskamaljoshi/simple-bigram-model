import json
import pytest
from serializer import Serializer


def test_serializer_save_and_load_valid(tmp_path):
    serializer = Serializer()
    model = {
        "<START>": {"the": 1.0},
        "the": {"cat": 0.5, "dog": 0.5},
        "cat": {"<END>": 1.0},
    }
    file_path = tmp_path / "model.json"

    serializer.save(model, str(file_path))
    assert file_path.exists()

    loaded = serializer.load(str(file_path))
    assert loaded == model


def test_serializer_load_validate_false(tmp_path):
    serializer = Serializer()
    model = {"hello": {"world": 1.0}}
    file_path = tmp_path / "model_fast.json"

    serializer.save(model, str(file_path))
    loaded = serializer.load(str(file_path), validate=False)
    assert loaded == model


def test_serializer_load_invalid_root_type(tmp_path):
    serializer = Serializer()
    file_path = tmp_path / "bad_root.json"
    file_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    with pytest.raises(ValueError, match="Saved model must contain a probability map"):
        serializer.load(str(file_path))


def test_serializer_load_invalid_word_key(tmp_path):
    serializer = Serializer()
    file_path = tmp_path / "bad_word.json"
    # When serialized as JSON, keys are always strings, but let's test invalid transition type
    file_path.write_text(json.dumps({"valid_word": [1, 2, 3]}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid probability map structure"):
        serializer.load(str(file_path))


def test_serializer_load_boolean_probability(tmp_path):
    serializer = Serializer()
    file_path = tmp_path / "bool_prob.json"
    file_path.write_text(json.dumps({"word": {"next": True}}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid transition probabilities"):
        serializer.load(str(file_path))


def test_serializer_load_non_numeric_probability(tmp_path):
    serializer = Serializer()
    file_path = tmp_path / "str_prob.json"
    file_path.write_text(json.dumps({"word": {"next": "high"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid transition probabilities"):
        serializer.load(str(file_path))
