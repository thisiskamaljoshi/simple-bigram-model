import math


class Evaluator:
    def validate_probabilities(self, model: dict[str, dict[str, float]]) -> None:
        epsilon = 0.001
        for word, transitions in model.items():
            total_probabilities = 0.0
            for next_word, probability in transitions.items():
                if 0.0 <= probability <= 1.0:
                    total_probabilities += probability
                else:
                    raise ValueError(f"Probability for ({word}, {next_word}) must be between 0 and 1, got {probability}")
            if abs(total_probabilities - 1.0) > epsilon:
                raise ValueError(f"Probabilities for '{word}' sum to {total_probabilities}, expected approximately 1.0")

    def vocabulary_size(self, model: dict[str, dict[str, float]]) -> int:
        vocab: set[str] = set()
        for word, transitions in model.items():
            vocab.add(word)
            vocab.update(transitions.keys())
        return len(vocab)

    def unique_bigram_count(self, model: dict[str, dict[str, float]]) -> int:
        return sum(len(transitions) for transitions in model.values())

    def get_bigrams(self, test_tokens: list[str]) -> list[tuple[str, str]]:
        if len(test_tokens) < 2:
            return []
        return list(zip(test_tokens[:-1], test_tokens[1:]))

    def get_probability(
        self,
        model: dict[str, dict[str, float]],
        current_word: str,
        next_word: str,
    ) -> float:
        if current_word in model:
            return model[current_word].get(next_word, 0.0)
        return 0.0

    def evaluate_seen_vs_unseen_bigrams(
        self,
        model: dict[str, dict[str, float]],
        test_tokens: list[str],
    ) -> dict[str, int]:
        total_bigrams = max(0, len(test_tokens) - 1)
        seen_bigrams_count = 0
        unseen_bigrams_count = 0

        for i in range(total_bigrams):
            first_word = test_tokens[i]
            second_word = test_tokens[i + 1]
            if first_word in model and second_word in model[first_word] and model[first_word][second_word] > 0.0:
                seen_bigrams_count += 1
            else:
                unseen_bigrams_count += 1

        return {
            "total_bigrams": total_bigrams,
            "seen_bigrams": seen_bigrams_count,
            "unseen_bigrams": unseen_bigrams_count,
        }

    def calculate_coverage(
        self,
        seen_bigrams: int,
        total_bigrams: int,
    ) -> float:
        if total_bigrams == 0:
            return 0.0
        return seen_bigrams / total_bigrams

    def calculate_log_probabilities(
        self,
        model: dict[str, dict[str, float]],
        test_tokens: list[str],
    ) -> dict[str, float]:
        if len(test_tokens) < 2:
            return {
                "total_log_probability": 0.0,
                "average_log_probability": 0.0,
            }

        total_bigrams = len(test_tokens) - 1
        total_log_probability = 0.0

        for i in range(total_bigrams):
            probability = self.get_probability(model, test_tokens[i], test_tokens[i + 1])
            if probability == 0.0:
                return {
                    "total_log_probability": -math.inf,
                    "average_log_probability": -math.inf,
                }
            total_log_probability += math.log(probability)

        average_log_probability = total_log_probability / total_bigrams

        return {
            "total_log_probability": total_log_probability,
            "average_log_probability": average_log_probability,
        }

    def calculate_perplexity(self, average_log_probability: float) -> float:
        if average_log_probability == -math.inf:
            return math.inf
        try:
            return math.exp(-average_log_probability)
        except OverflowError:
            return math.inf

    def evaluate(
        self,
        model: dict[str, dict[str, float]],
        test_tokens: list[str],
    ) -> dict[str, int | float]:
        self.validate_probabilities(model)

        if len(test_tokens) < 2:
            return {
                "vocabulary_size": self.vocabulary_size(model),
                "unique_bigram_count": self.unique_bigram_count(model),
                "evaluated_bigrams": 0,
                "seen_bigrams": 0,
                "unseen_bigrams": 0,
                "coverage": 0.0,
                "total_log_probability": 0.0,
                "average_log_probability": 0.0,
                "perplexity": 1.0,
            }

        log_probabilities = self.calculate_log_probabilities(model, test_tokens)
        bigrams = self.evaluate_seen_vs_unseen_bigrams(model, test_tokens)

        return {
            "vocabulary_size": self.vocabulary_size(model),
            "unique_bigram_count": self.unique_bigram_count(model),
            "evaluated_bigrams": bigrams["total_bigrams"],
            "seen_bigrams": bigrams["seen_bigrams"],
            "unseen_bigrams": bigrams["unseen_bigrams"],
            "coverage": self.calculate_coverage(
                bigrams["seen_bigrams"],
                bigrams["total_bigrams"],
            ),
            "total_log_probability": log_probabilities["total_log_probability"],
            "average_log_probability": log_probabilities["average_log_probability"],
            "perplexity": self.calculate_perplexity(log_probabilities["average_log_probability"]),
        }

    def report(self, metrics: dict[str, int | float]) -> None:
        print("===== Bigram Model Evaluation =====")
        print("Vocabulary size:", metrics["vocabulary_size"])
        print("Unique learned bigrams:", metrics["unique_bigram_count"])
        print("Evaluated bigrams:", metrics["evaluated_bigrams"])
        print("Seen bigrams:", metrics["seen_bigrams"])
        print("Unseen bigrams:", metrics["unseen_bigrams"])
        print("Coverage:", f"{metrics['coverage']:.2%}")
        print("Total log probability:", metrics["total_log_probability"])
        print("Average log probability:", metrics["average_log_probability"])
        print("Perplexity:", metrics["perplexity"])

