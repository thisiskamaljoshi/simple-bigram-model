import json
from pathlib import Path


class Serializer:
    """Save and load trained bigram probability maps as JSON files."""

    def save(self, model: dict[str, dict[str, float]], file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(model, file, ensure_ascii=False, separators=(",", ":"))

    def load(self, file_path: str) -> dict[str, dict[str, float]]:
        path = Path(file_path)

        with path.open("r", encoding="utf-8") as file:
            model = json.load(file)

        if not isinstance(model, dict):
            raise ValueError("Saved model must contain a probability map.")

        for word, transitions in model.items():
            if not isinstance(word, str) or not isinstance(transitions, dict):
                raise ValueError("Saved model has an invalid probability map structure.")
            if any(
                not isinstance(next_word, str)
                or isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                for next_word, probability in transitions.items()
            ):
                raise ValueError("Saved model contains invalid transition probabilities.")

        return model
