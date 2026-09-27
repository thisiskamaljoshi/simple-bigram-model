import argparse
import random
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


def train_model(corpus_path: str | None = None) -> None:
    corpus = Corpus()
    tokenizer = Tokenizer()
    sentence_processor = SentenceProcessor()
    model_trainer = Trainer()
    bigram_model = BigramModel()
    serializer = Serializer()

    target_corpus = corpus_path or TRAIN_CORPUS_PATH
    print(f"Loading training corpus from: {target_corpus}")
    text = corpus.load(target_corpus)
    tokens = tokenizer.tokenize(text)
    processed_tokens = sentence_processor.process(tokens)
    trained_model = model_trainer.train(processed_tokens)
    bigram_probabilities = bigram_model.train(trained_model)
    serializer.save(bigram_probabilities, MODEL_PATH)

    print(f"Saved model to: {MODEL_PATH}")


def generate_text(
    temperature: float = 1.0,
    max_tokens: int = 100,
    seed: int | None = None,
) -> None:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"No saved model found at: {MODEL_PATH}")
        print("Train one first with: python main.py train")
        return

    if seed is not None:
        random.seed(seed)

    serializer = Serializer()
    generator = Generator()
    bigram_probabilities = serializer.load(MODEL_PATH, validate=False)

    generated_output = generator.generate_sentence(
        bigram_probabilities,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    print(generated_output)


def evaluate_model() -> None:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"No saved model found at: {MODEL_PATH}")
        print("Train one first with: python main.py train")
        return

    test_corpus_path = Path(TEST_CORPUS_PATH)
    if not test_corpus_path.exists():
        print(f"No saved test corpus found at: {TEST_CORPUS_PATH}")
        print("Generate a test corpus first with: python download_corpus.py")
        return

    serializer = Serializer()
    corpus = Corpus()
    tokenizer = Tokenizer()
    sentence_processor = SentenceProcessor()
    evaluator = Evaluator()

    model = serializer.load(MODEL_PATH, validate=False)
    test_text = corpus.load(TEST_CORPUS_PATH)

    tokens = tokenizer.tokenize(test_text)
    processed_tokens = sentence_processor.process(tokens)
    metrics = evaluator.evaluate(model, processed_tokens)
    evaluator.report(metrics)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train, generate text, or evaluate the statistical bigram model."
    )
    parser.add_argument(
        "command",
        choices=("train", "generate", "evaluate"),
        help="Command to execute: train, generate, or evaluate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (lower = more deterministic/greedy, higher = more random). Default: 1.0",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate. Default: 100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible text generation.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Custom corpus file path for training.",
    )

    args = parser.parse_args(argv)

    if args.command == "train":
        train_model(corpus_path=args.corpus)
    elif args.command == "generate":
        generate_text(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    elif args.command == "evaluate":
        evaluate_model()


if __name__ == "__main__":
    main()

