from corpus import Corpus
from tokenizer import Tokenizer
from sentenceprocessor import SentenceProcessor
from trainer import Trainer
from bigram import BigramModel

def main():
    corpus = Corpus()
    tokenizer = Tokenizer()
    sentence_processor = SentenceProcessor()
    model_trainer = Trainer()
    bigram_model = BigramModel()

    text = corpus.load("corpus.txt")
    tokens = tokenizer.tokenize(text)
    processed_tokens = sentence_processor.process(tokens)
    trained_model = model_trainer.train(processed_tokens)
    bigram_probabilities = bigram_model.train(trained_model)

    print("Corpus:")
    print(bigram_probabilities)


if __name__ == "__main__":
    main()