from corpus import Corpus
from tokenizer import Tokenizer
from sentenceprocessor import SentenceProcessor
from trainer import Trainer
from bigram import BigramModel
from generator import Generator

def main():
    corpus = Corpus()
    tokenizer = Tokenizer()
    sentence_processor = SentenceProcessor()
    model_trainer = Trainer()
    bigram_model = BigramModel()
    generate = Generator()

    text = corpus.load("dataset/tinystories/tinystories_50mb.txt")
    tokens = tokenizer.tokenize(text)
    processed_tokens = sentence_processor.process(tokens)
    trained_model = model_trainer.train(processed_tokens)
    bigram_probabilities = bigram_model.train(trained_model)

    generated_output = generate.generate_sentence(bigram_probabilities)

    print(generated_output)


if __name__ == "__main__":
    main()
