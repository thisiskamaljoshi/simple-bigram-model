import random
class Sampler:
    '''
    Randomly selects the next word based on probabilities.

    Probabilistic sampling:
    Pick randomly, but give more likely words a higher chance.
    '''

    def sample( self, probabilities: dict[str, float] ) -> str | None:
        random_value = random.random()

        cumulative_probability = 0
        for word in probabilities:
            cumulative_probability += probabilities[word]
            if random_value > cumulative_probability:
                continue
            else:
                return word

        return None
                

