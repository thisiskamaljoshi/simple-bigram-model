import sys
from pathlib import Path
from unittest.mock import patch

from main import train_model, generate_text, evaluate_model, main


def test_generate_text_missing_model(capsys, monkeypatch):
    monkeypatch.setattr("main.MODEL_PATH", "non_existent_model_file_123.json")
    generate_text()
    captured = capsys.readouterr()
    assert "No saved model found" in captured.out


def test_evaluate_model_missing_model(capsys, monkeypatch):
    monkeypatch.setattr("main.MODEL_PATH", "non_existent_model_file_123.json")
    evaluate_model()
    captured = capsys.readouterr()
    assert "No saved model found" in captured.out


def test_evaluate_model_missing_test_corpus(capsys, monkeypatch, tmp_path):
    # Model exists, but test corpus does not
    model_file = tmp_path / "model.json"
    model_file.write_text('{"<START>": {"hello": 1.0}}', encoding="utf-8")
    monkeypatch.setattr("main.MODEL_PATH", str(model_file))
    monkeypatch.setattr("main.TEST_CORPUS_PATH", str(tmp_path / "non_existent_test.txt"))

    evaluate_model()
    captured = capsys.readouterr()
    assert "No saved test corpus found" in captured.out


def test_train_and_generate_end_to_end(tmp_path, capsys, monkeypatch):
    train_file = tmp_path / "tiny_train.txt"
    train_file.write_text("The little cat sat on the mat. The dog barked.", encoding="utf-8")
    model_file = tmp_path / "tiny_model.json"

    monkeypatch.setattr("main.MODEL_PATH", str(model_file))
    monkeypatch.setattr("main.TRAIN_CORPUS_PATH", str(train_file))

    # Train model
    train_model(corpus_path=str(train_file))
    captured = capsys.readouterr()
    assert "Saved model to:" in captured.out
    assert model_file.exists()

    # Generate text
    generate_text(temperature=0.01, seed=42)
    captured = capsys.readouterr()
    assert len(captured.out.strip()) > 0


def test_main_cli_dispatch(monkeypatch):
    with patch("main.train_model") as mock_train:
        monkeypatch.setattr(sys, "argv", ["main.py", "train", "--corpus", "custom.txt"])
        main()
        mock_train.assert_called_once_with(corpus_path="custom.txt")

    with patch("main.generate_text") as mock_gen:
        monkeypatch.setattr(sys, "argv", ["main.py", "generate", "--temperature", "0.5", "--max-tokens", "50", "--seed", "123"])
        main()
        mock_gen.assert_called_once_with(temperature=0.5, max_tokens=50, seed=123)

    with patch("main.evaluate_model") as mock_eval:
        monkeypatch.setattr(sys, "argv", ["main.py", "evaluate"])
        main()
        mock_eval.assert_called_once()


def test_evaluate_model_success(tmp_path, capsys, monkeypatch):
    train_file = tmp_path / "tiny_train.txt"
    train_file.write_text("The little cat sat on the mat.", encoding="utf-8")
    test_file = tmp_path / "tiny_test.txt"
    test_file.write_text("The little cat.", encoding="utf-8")
    model_file = tmp_path / "tiny_model.json"

    monkeypatch.setattr("main.MODEL_PATH", str(model_file))
    monkeypatch.setattr("main.TRAIN_CORPUS_PATH", str(train_file))
    monkeypatch.setattr("main.TEST_CORPUS_PATH", str(test_file))

    train_model(corpus_path=str(train_file))
    evaluate_model()

    captured = capsys.readouterr()
    assert "===== Bigram Model Evaluation =====" in captured.out
    assert "Coverage:" in captured.out


def test_main_argv_argument():
    with patch("main.generate_text") as mock_gen:
        main(["generate", "--temperature", "0.2"])
        mock_gen.assert_called_once_with(temperature=0.2, max_tokens=100, seed=None)


