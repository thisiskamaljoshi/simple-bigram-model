import os
from datasets import load_dataset
from config import CORPORA


class CorpusDownloader:

    def download(self, corpus_name: str):

        if corpus_name not in CORPORA:
            print("Corpus not found.")
            return

        corpus = CORPORA[corpus_name]

        dataset_name = corpus["dataset_name"]
        split = corpus["split"]
        target_size_mb = corpus["target_size_mb"]
        output_file = corpus["output_file"]

        target_bytes = target_size_mb * 1024 * 1024

        # Check if the file already exists
        if os.path.exists(output_file):

            file_size = os.path.getsize(output_file)
            file_size_mb = file_size / (1024 * 1024)

            if file_size >= target_bytes:
                print(f"{corpus_name} already exists.")
                print(f"Current size: {file_size_mb:.2f} MB")
                print("Download skipped.")
                return

        written_bytes = 0

        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True
        )

        os.makedirs(
            os.path.dirname(output_file),
            exist_ok=True
        )

        with open(output_file, "w", encoding="utf-8") as file:

            for item in dataset:

                text = item["text"].strip() + "\n"

                file.write(text)

                written_bytes += len(text.encode("utf-8"))

                if written_bytes >= target_bytes:
                    break

        print(f"Downloaded: {corpus_name}")
        print(f"Created: {output_file}")

if __name__ == "__main__":
    downloader = CorpusDownloader()
    downloader.download("tinystories_50mb")