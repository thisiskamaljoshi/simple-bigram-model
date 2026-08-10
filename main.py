from corpus import Corpus
from tokenizer import Tokenizer

def main():
    corpus = Corpus()

    text = corpus.load("corpus.txt")

    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)

    print("Corpus:")
    print(tokens)


if __name__ == "__main__":
    main()