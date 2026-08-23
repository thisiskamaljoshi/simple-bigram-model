from helpers import NO_SPACE_BEFORE,NO_SPACE_AFTER,NO_SPACE_AROUND

class Detokenizer:
    '''
    Converts a list of tokens back into readable text.
    '''
    def process(self, tokens: list[str]) -> str:
        sentence = ''
        nospaceafter = False
        for token in tokens:
            if token in NO_SPACE_BEFORE or nospaceafter:
                sentence = sentence + token
                nospaceafter = False
            elif token in NO_SPACE_AFTER:
                sentence = sentence + " " + token
                nospaceafter = True
            elif token in NO_SPACE_AROUND:
                sentence = sentence + token
                nospaceafter = True
            else:
                sentence = sentence + " " + token

        return sentence
