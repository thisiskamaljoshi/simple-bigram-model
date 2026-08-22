class BigramModel:
    '''
    Converts bigram counts into conditional probabilities.
    '''
    def train(self ,bigram_count: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
        bigram_probabilities = {}
        for word in bigram_count:
            count = 0
            bigram_probabilities[word] = {}
            for next_word in bigram_count[word]:
                count = count + bigram_count[word][next_word]
            for next_word in bigram_count[word]:
                bigram_probabilities[word][next_word] = bigram_count[word][next_word]/ count
            
