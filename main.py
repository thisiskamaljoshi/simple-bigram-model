import argparse
from pathlib import Path

from corpus import Corpus
from tokenizer import Tokenizer
from sentenceprocessor import SentenceProcessor
from trainer import Trainer
from bigram import BigramModel
from generator import Generator
from serializer import Serializer
from evaluator import Evaluator
from config import MODEL_PATH, TRAIN_CORPUS_PATH, TEST_CORPUS_PATH

def train_model() -> None:
    corpus = Corpus()
    tokenizer = Tokenizer()
    sentence_processor = SentenceProcessor()
    model_trainer = Trainer()
    bigram_model = BigramModel()
    serializer = Serializer()

    text = corpus.load(TRAIN_CORPUS_PATH)
    tokens = tokenizer.tokenize(text)
    processed_tokens = sentence_processor.process(tokens)
    trained_model = model_trainer.train(processed_tokens)
    bigram_probabilities = bigram_model.train(trained_model)
    serializer.save(bigram_probabilities, MODEL_PATH)

    print(f"Saved model to: {MODEL_PATH}")


def generate_text() -> None:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"No saved model found at: {MODEL_PATH}")
        print("Train one first with: python main.py train")
        return

    serializer = Serializer()
    generate = Generator()
    bigram_probabilities = serializer.load(MODEL_PATH)

    generated_output = generate.generate_sentence(bigram_probabilities)
    print(generated_output)

def evaluate_model()->None:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"No saved model found at: {MODEL_PATH}")
        print("Train one first with: python main.py train")
        return

    test_corpus_path = Path(TEST_CORPUS_PATH)
    if not test_corpus_path.exists():
        print(f"No saved test corpus found at: {TEST_CORPUS_PATH}")
        print("Generate a test corpus first with: python main.py train")
        return

    serializer = Serializer()
    corpus = Corpus()
    tokenizer = Tokenizer()
    sentence_processor = SentenceProcessor()
    evaluator = Evaluator()

    model = serializer.load(MODEL_PATH)
    test_text = corpus.load(TEST_CORPUS_PATH)
    
    tokens = tokenizer.tokenize(test_text)
    
    processed_tokens = sentence_processor.process(tokens)
    metrics = evaluator.evaluate(model, processed_tokens)
    evaluator.report(metrics)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or generate text with the bigram model."
    )
    parser.add_argument("command", choices=("train", "generate", "evaluate"))
    args = parser.parse_args()

    if args.command == "train":
        train_model()
    elif args.command == "generate":
        generate_text()
    elif args.command == "evaluate":
        evaluate_model()


if __name__ == "__main__":
    main()
