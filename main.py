from corpus import Corpus
from tokenizer import Tokenizer
from sentenceprocessor import SentenceProcessor

def main():
    corpus = Corpus()

    text = corpus.load("corpus.txt")

    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    sentence_processor = SentenceProcessor()
    processed_tokens = sentence_processor.process(tokens)

    print("Corpus:")
    print(processed_tokens)


if __name__ == "__main__":
    main()