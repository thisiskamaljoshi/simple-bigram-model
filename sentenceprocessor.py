class SentenceProcessor:
    def process(self, tokens: list[str])->list[str]:
        sentence_endings  = {".","?","!"}
        processed_tokens = ["<START>"]
        for i, token in enumerate(tokens):
            if token not in sentence_endings:
                processed_tokens.append(token)
            else:
                processed_tokens.append(token)
                processed_tokens.append("<END>")
                if i+1 < len(tokens):
                    processed_tokens.append("<START>")
        return processed_tokens

        

