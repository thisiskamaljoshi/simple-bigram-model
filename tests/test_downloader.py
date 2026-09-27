from pathlib import Path
from download_corpus import CorpusDownloader


def test_is_test_story_deterministic():
    story1 = "Once upon a time, there was a little bird named Pip."
    result1 = CorpusDownloader._is_test_story(story1)
    result2 = CorpusDownloader._is_test_story(story1)
    # Must be 100% deterministic
    assert result1 == result2


def test_is_test_story_distribution():
    # Over 1000 sample stories, test ratio should be approximately 20% (+/- 5%)
    test_count = 0
    total = 1000
    for i in range(total):
        story = f"This is synthetic unique test story number {i} for hashing."
        if CorpusDownloader._is_test_story(story):
            test_count += 1

    ratio = test_count / total
    assert 0.14 <= ratio <= 0.26, f"Expected ~0.20 split ratio, got {ratio}"


def test_file_size(tmp_path):
    f = tmp_path / "dummy.txt"
    f.write_text("Hello, 12345", encoding="utf-8")
    assert CorpusDownloader._file_size(f) == len("Hello, 12345".encode("utf-8"))


def test_print_split_summary(tmp_path, capsys):
    train_file = tmp_path / "train.txt"
    test_file = tmp_path / "test.txt"
    train_file.write_text("Train content 123", encoding="utf-8")
    test_file.write_text("Test content 123", encoding="utf-8")

    CorpusDownloader._print_split_summary(train_file, test_file, train_stories=10, test_stories=2)
    captured = capsys.readouterr()
    assert "Training stories: 10" in captured.out
    assert "Test stories: 2" in captured.out


def test_download_unknown_corpus(capsys):
    downloader = CorpusDownloader()
    downloader.download("non_existent_corpus_xyz")
    captured = capsys.readouterr()
    assert "Corpus not found." in captured.out


def test_download_already_exists(tmp_path, capsys, monkeypatch):
    train_file = tmp_path / "train.txt"
    test_file = tmp_path / "test.txt"
    train_file.write_text("dummy training content" * 10, encoding="utf-8")
    test_file.write_text("dummy test content" * 10, encoding="utf-8")

    mock_corpora = {
        "test_corpus": {
            "dataset_name": "dummy_dataset",
            "split": "train",
            "target_size_mb": 0.00001,  # ~10 bytes
            "train_output_file": str(train_file),
            "test_output_file": str(test_file),
        }
    }
    monkeypatch.setattr("download_corpus.CORPORA", mock_corpora)

    downloader = CorpusDownloader()
    downloader.download("test_corpus")

    captured = capsys.readouterr()
    assert "split already exists." in captured.out
    assert "Download skipped." in captured.out


def test_download_streaming_success(tmp_path, capsys, monkeypatch):
    train_file = tmp_path / "train.txt"
    test_file = tmp_path / "test.txt"

    mock_corpora = {
        "test_corpus": {
            "dataset_name": "dummy_dataset",
            "split": "train",
            "target_size_mb": 0.0001,
            "train_output_file": str(train_file),
            "test_output_file": str(test_file),
        }
    }
    monkeypatch.setattr("download_corpus.CORPORA", mock_corpora)

    mock_items = [
        {"text": "Once upon a time, there was a little bird."},
        {"text": "Another story with lots of words to meet target size."},
        {"text": ""},  # empty story branch
    ]

    monkeypatch.setattr("download_corpus.load_dataset", lambda *args, **kwargs: mock_items)

    downloader = CorpusDownloader()
    downloader.download("test_corpus")

    assert train_file.exists() or test_file.exists()
    captured = capsys.readouterr()
    assert "Downloaded: test_corpus" in captured.out


def test_download_exception_cleanup(tmp_path, monkeypatch):
    train_file = tmp_path / "train.txt"
    test_file = tmp_path / "test.txt"

    mock_corpora = {
        "test_corpus": {
            "dataset_name": "dummy_dataset",
            "split": "train",
            "target_size_mb": 1.0,
            "train_output_file": str(train_file),
            "test_output_file": str(test_file),
        }
    }
    monkeypatch.setattr("download_corpus.CORPORA", mock_corpora)

    def failing_dataset(*args, **kwargs):
        def error_gen():
            yield {"text": "Valid story"}
            raise RuntimeError("Network failed mid-download")
        return error_gen()

    monkeypatch.setattr("download_corpus.load_dataset", failing_dataset)

    downloader = CorpusDownloader()
    import pytest
    with pytest.raises(RuntimeError, match="Network failed mid-download"):
        downloader.download("test_corpus")

    # Temp files must be cleanly deleted
    assert not (tmp_path / "train.tmp").exists()
    assert not (tmp_path / "test.tmp").exists()

