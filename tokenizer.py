import string

PUNCTUATION = set(string.punctuation)

class Tokenizer:
    def tokenize(self, text: str) -> list[str]:
        tokens = []
        word_buffer = ""
        
        for char in text:
            if char.isspace():
                if word_buffer != "":
                    tokens.append(word_buffer)
                    word_buffer = ""
            elif char not in PUNCTUATION:
                word_buffer = word_buffer + char
            else:
                if word_buffer != "":
                    tokens.append(word_buffer)
                tokens.append(char)
                word_buffer = ""

        if word_buffer != "":
            tokens.append(word_buffer)

        return tokens