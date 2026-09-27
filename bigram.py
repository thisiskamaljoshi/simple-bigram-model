class BigramModel:
    '''
    Converts bigram counts into conditional probabilities using Maximum Likelihood Estimation (MLE).
    '''
    def train(self, bigram_count: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
        bigram_probabilities: dict[str, dict[str, float]] = {}

        for word, next_counts in bigram_count.items():
            total_count = sum(next_counts.values())
            if total_count == 0:
                continue

            bigram_probabilities[word] = {
                next_word: count / total_count
                for next_word, count in next_counts.items()
            }

        return bigram_probabilities

            
