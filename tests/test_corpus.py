import pytest
from corpus import Corpus


def test_corpus_load_valid(tmp_path):
    corpus = Corpus()
    file_path = tmp_path / "test.txt"
    content = "Once upon a time, there was a little bird."
    file_path.write_text(content, encoding="utf-8")

    loaded = corpus.load(str(file_path))
    assert loaded == content


def test_corpus_load_utf8_bom_stripping(tmp_path):
    corpus = Corpus()
    file_path = tmp_path / "bom.txt"
    # Write with explicit UTF-8 BOM bytes (\xef\xbb\xbf)
    file_path.write_bytes(b"\xef\xbb\xbfOnce upon a time.")

    loaded = corpus.load(str(file_path))
    assert loaded == "Once upon a time."
    assert not loaded.startswith("\ufeff")


def test_corpus_load_file_not_found():
    corpus = Corpus()
    with pytest.raises(FileNotFoundError, match="Corpus file not found"):
        corpus.load("non_existent_file_path_12345.txt")
