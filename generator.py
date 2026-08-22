class Generator:
    '''
    Generate next word
    '''
    def generate(self, bigram_probabilities, current_word) -> str | None:
        if current_word in bigram_probabilities:
            max_probability = 0
            max_probability_word = ''
            for next_word in bigram_probabilities[current_word]:
                if(bigram_probabilities[current_word][next_word] > max_probability):
                    max_probability = bigram_probabilities[current_word][next_word]
                    max_probability_word = next_word
            return max_probability_word
        else:
            return None

    def generate_sentence(self, bigram_probabilities) -> str:
        generated_sentence = ''
        current_word = '<START>'
        for i in range(0,100):
            generated_word =  self.generate(bigram_probabilities, current_word)
            if generated_word == None:
                break
            if generated_word == '<END>':
                break
            generated_sentence = generated_sentence + " " + generated_word
            current_word = generated_word
        return generated_sentence
            

