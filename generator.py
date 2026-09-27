from sampler import Sampler
from detokenizer import Detokenizer


class Generator:
    '''
    Autoregressively generates sentences using learned bigram transition probabilities.
    '''
    def __init__(self) -> None:
        self.sampler = Sampler()
        self.detokenizer = Detokenizer()

    def generate(
        self,
        bigram_probabilities: dict[str, dict[str, float]],
        current_word: str,
        temperature: float = 1.0,
    ) -> str | None:
        if current_word in bigram_probabilities:
            return self.sampler.sample(bigram_probabilities[current_word], temperature=temperature)
        return None

    def generate_sentence(
        self,
        bigram_probabilities: dict[str, dict[str, float]],
        max_tokens: int = 100,
        temperature: float = 1.0,
    ) -> str:
        generated_tokens = []
        current_word = '<START>'

        for _ in range(max_tokens):
            generated_word = self.generate(
                bigram_probabilities, current_word, temperature=temperature
            )
            if generated_word is None or generated_word == '<END>':
                break
            generated_tokens.append(generated_word)
            current_word = generated_word

        return self.detokenizer.process(generated_tokens)

            

