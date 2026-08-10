from pathlib import Path


class Corpus:
    def load(self, file_path: str) -> str:
        """
        Load the contents of a text file and return it as a string.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {file_path}")

        return path.read_text(encoding="utf-8")