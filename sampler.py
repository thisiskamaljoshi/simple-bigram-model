import math
import random


class Sampler:
    '''
    Randomly selects the next word based on probabilities.

    Probabilistic sampling:
    Pick randomly, but give more likely words a higher chance.
    Supports temperature scaling and is robust against float roundoff.
    '''

    def sample(
        self,
        probabilities: dict[str, float],
        temperature: float = 1.0,
    ) -> str | None:
        if not probabilities:
            return None

        words = list(probabilities.keys())
        probs = list(probabilities.values())

        # Greedy / Argmax decoding for temperature near 0
        if temperature <= 0.05:
            max_idx = probs.index(max(probs))
            return words[max_idx]

        # Temperature scaling
        if temperature != 1.0:
            scaled_probs = [math.exp(math.log(max(p, 1e-12)) / temperature) for p in probs]
            total = sum(scaled_probs)
            if total > 0:
                probs = [p / total for p in scaled_probs]

        try:
            return random.choices(words, weights=probs, k=1)[0]
        except (ValueError, IndexError):
            return words[probs.index(max(probs))]
                

