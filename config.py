CORPORA = {
    "tinystories_50mb": {
        "dataset_name": "roneneldan/TinyStories",
        "split": "train",
        "target_size_mb": 50,
        "train_output_file": "dataset/tinystories/tinystories_50mb_train.txt",
        "test_output_file": "dataset/tinystories/tinystories_50mb_test.txt",
    },

    "tinystories_100mb": {
        "dataset_name": "roneneldan/TinyStories",
        "split": "train",
        "target_size_mb": 100,
        "train_output_file": "dataset/tinystories/tinystories_100mb_train.txt",
        "test_output_file": "dataset/tinystories/tinystories_100mb_test.txt",
    },
}

ACTIVE_CORPUS = "tinystories_50mb"

TRAIN_CORPUS_PATH = CORPORA[ACTIVE_CORPUS]["train_output_file"]
TEST_CORPUS_PATH = CORPORA[ACTIVE_CORPUS]["test_output_file"]
MODEL_PATH = f"models/{ACTIVE_CORPUS}_bigrams.json"