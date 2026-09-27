from config import CORPORA, TRAIN_CORPUS_PATH, TEST_CORPUS_PATH, MODEL_PATH


def test_config_corpora_structure():
    assert "tinystories_50mb" in CORPORA
    assert "tinystories_100mb" in CORPORA

    for name, spec in CORPORA.items():
        assert "dataset_name" in spec
        assert "split" in spec
        assert "target_size_mb" in spec
        assert "train_output_file" in spec
        assert "test_output_file" in spec
        assert isinstance(spec["target_size_mb"], int)


def test_config_paths():
    assert isinstance(TRAIN_CORPUS_PATH, str) and len(TRAIN_CORPUS_PATH) > 0
    assert isinstance(TEST_CORPUS_PATH, str) and len(TEST_CORPUS_PATH) > 0
    assert isinstance(MODEL_PATH, str) and len(MODEL_PATH) > 0
