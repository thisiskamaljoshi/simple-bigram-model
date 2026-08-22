class Trainer:
    '''
    Scans the processed token sequence, finds every pair of adjacent
    tokens, and counts how often each token is followed by another token.
    '''
    def train(self,tokens: list[str]) -> dict[str, dict[str, int]]:
        bigram_dict = {}

        for i,token in enumerate(tokens):
            if i < len(tokens)-1:
                if token in bigram_dict:
                    if tokens[i+1] in bigram_dict[token]:
                        bigram_dict[token][tokens[i+1]] = bigram_dict[token][tokens[i+1]] + 1
                    else:
                        bigram_dict[token][tokens[i+1]] = 1 
                else:
                    bigram_dict[token] = { tokens[i+1] : 1}

        return bigram_dict