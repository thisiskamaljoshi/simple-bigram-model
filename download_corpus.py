import hashlib
import os
from pathlib import Path

from datasets import load_dataset
from config import CORPORA


class CorpusDownloader:
    """Download TinyStories and create a deterministic 80/20 story split."""

    TEST_PERCENTAGE = 20

    @staticmethod
    def _file_size(path: Path) -> int:
        return path.stat().st_size

    @classmethod
    def _is_test_story(cls, text: str) -> bool:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big") % 100
        return bucket < cls.TEST_PERCENTAGE

    @staticmethod
    def _print_split_summary(
        train_path: Path,
        test_path: Path,
        train_stories: int | None = None,
        test_stories: int | None = None,
    ) -> None:
        train_bytes = train_path.stat().st_size
        test_bytes = test_path.stat().st_size
        total_bytes = train_bytes + test_bytes
        train_percentage = (train_bytes / total_bytes * 100) if total_bytes else 0
        test_percentage = (test_bytes / total_bytes * 100) if total_bytes else 0

        if train_stories is not None and test_stories is not None:
            print(f"Training stories: {train_stories}")
            print(f"Test stories: {test_stories}")
        print(f"Training corpus: {train_bytes / (1024 * 1024):.2f} MB ({train_percentage:.1f}%)")
        print(f"Test corpus: {test_bytes / (1024 * 1024):.2f} MB ({test_percentage:.1f}%)")
        print(f"Combined corpus: {total_bytes / (1024 * 1024):.2f} MB")

    def download(self, corpus_name: str):
        if corpus_name not in CORPORA:
            print("Corpus not found.")
            return

        corpus = CORPORA[corpus_name]

        dataset_name = corpus["dataset_name"]
        split = corpus["split"]
        target_size_mb = corpus["target_size_mb"]
        train_path = Path(corpus["train_output_file"])
        test_path = Path(corpus["test_output_file"])

        target_bytes = target_size_mb * 1024 * 1024

        if train_path.exists() and test_path.exists():
            combined_bytes = self._file_size(train_path) + self._file_size(test_path)
            if combined_bytes >= target_bytes:
                print(f"{corpus_name} split already exists.")
                self._print_split_summary(train_path, test_path)
                print("Download skipped.")
                return

        if train_path.exists() or test_path.exists():
            print("Incomplete corpus split found. Recreating both files.")

        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True
        )

        train_path.parent.mkdir(parents=True, exist_ok=True)
        temp_train_path = train_path.with_suffix(".tmp")
        temp_test_path = test_path.with_suffix(".tmp")
        written_bytes = 0
        train_stories = 0
        test_stories = 0

        try:
            with temp_train_path.open("w", encoding="utf-8") as train_file, temp_test_path.open(
                "w", encoding="utf-8"
            ) as test_file:
                for item in dataset:
                    story = " ".join(item["text"].split())
                    if not story:
                        continue

                    encoded_story = f"{story}\n".encode("utf-8")
                    if self._is_test_story(story):
                        test_file.write(encoded_story.decode("utf-8"))
                        test_stories += 1
                    else:
                        train_file.write(encoded_story.decode("utf-8"))
                        train_stories += 1

                    written_bytes += len(encoded_story)
                    if written_bytes >= target_bytes:
                        break

            os.replace(temp_train_path, train_path)
            os.replace(temp_test_path, test_path)
        except Exception:
            temp_train_path.unlink(missing_ok=True)
            temp_test_path.unlink(missing_ok=True)
            raise

        print(f"Downloaded: {corpus_name}")
        self._print_split_summary(train_path, test_path, train_stories, test_stories)

if __name__ == "__main__":
    downloader = CorpusDownloader()
    downloader.download("tinystories_50mb")
