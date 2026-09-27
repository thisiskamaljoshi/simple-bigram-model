from helpers import NO_SPACE_AFTER, NO_SPACE_AROUND, NO_SPACE_BEFORE


class Detokenizer:
    '''
    Converts a list of tokens back into readable text.
    '''
    def process(self, tokens: list[str]) -> str:
        if not tokens:
            return ""

        output: list[str] = []
        nospaceafter = False

        for token in tokens:
            if not output:
                output.append(token)
                if token in NO_SPACE_AFTER or token in NO_SPACE_AROUND:
                    nospaceafter = True
            elif token in NO_SPACE_BEFORE or nospaceafter:
                output.append(token)
                nospaceafter = False
            elif token in NO_SPACE_AFTER:
                output.append(" " + token)
                nospaceafter = True
            elif token in NO_SPACE_AROUND:
                output.append(token)
                nospaceafter = True
            else:
                output.append(" " + token)

        return "".join(output)
