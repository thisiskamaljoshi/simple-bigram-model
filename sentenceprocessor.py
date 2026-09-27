class SentenceProcessor:
    '''
    Injects sentence boundary tokens (<START> and <END>) to demarcate sentences,
    robustly handling consecutive punctuation and boundary edge cases.
    '''
    def process(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []

        sentence_endings = {".", "?", "!"}
        processed: list[str] = ["<START>"]
        has_content = False
        n = len(tokens)

        for i, token in enumerate(tokens):
            if token not in sentence_endings:
                processed.append(token)
                has_content = True
            else:
                if not has_content:
                    # Skip orphan leading sentence endings before any words
                    continue

                processed.append(token)
                # Check if the next token is also a sentence ending
                is_last_token = (i + 1 == n)
                next_is_ending = not is_last_token and tokens[i + 1] in sentence_endings
                if not next_is_ending:
                    processed.append("<END>")
                    has_content = False
                    if not is_last_token:
                        processed.append("<START>")

        # Ensure trailing <END> if tokens ended without a sentence terminator
        if processed[-1] != "<END>":
            if has_content:
                processed.append("<END>")
            elif len(processed) > 1 and processed[-1] == "<START>":
                processed.pop()

        return processed if len(processed) > 1 else []


        

