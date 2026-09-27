from collections import defaultdict


class Trainer:
    '''
    Scans the processed token sequence, finds every pair of adjacent
    tokens, and counts how often each token is followed by another token.
    '''
    def train(self, tokens: list[str]) -> dict[str, dict[str, int]]:
        bigram_dict: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_transitions = len(tokens) - 1

        for i in range(total_transitions):
            bigram_dict[tokens[i]][tokens[i + 1]] += 1

        return {k: dict(v) for k, v in bigram_dict.items()}