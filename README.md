# Statistical Bigram Language Model

A small language model built from scratch in Python. It learns which words tend to follow one another in the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, then generates text by sampling from those learned probabilities.

This is a learning project, not a modern large language model. Its purpose is to make the core ideas behind statistical language modelling tangible: learn patterns from text, estimate probabilities, and predict what comes next.

## How it works

1. Download a text corpus from TinyStories.
2. Split the corpus into words and punctuation tokens.
3. Mark sentence beginnings and endings with `<START>` and `<END>` tokens.
4. Count adjacent word pairs (bigrams), such as `the -> cat`.
5. Convert counts into conditional next-word probabilities.
6. Generate a sentence by repeatedly sampling the next word from the probability map.

The trained probability map is saved as JSON, so generation does not need to retrain on the corpus every time.

## Features

- Streams and saves a 50 MB TinyStories training corpus
- Tokenizes text and preserves sentence boundaries
- Trains a word-level bigram probability model
- Saves trained probabilities to a reusable JSON model file
- Generates a new sentence each time through probabilistic sampling

## Setup

Requirements: Python 3.11 or newer and an internet connection for downloading dependencies and the corpus.

```powershell
git clone <your-repository-url>
cd simple-bigram-model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If your network uses a custom Windows certificate and pip reports an SSL verification error, retry the final command with:

```powershell
python -m pip install --use-feature=truststore -r requirements.txt
```

## Usage

Download the configured TinyStories corpus. This is only needed once unless you remove the dataset file.

```powershell
python download_corpus.py
```

Train the model. This reads the corpus and saves the learned probability map to `models/tinystories_50mb_bigrams.json`.

```powershell
python main.py train
```

Generate text from the saved model. This does **not** read the corpus or retrain the model.

```powershell
python main.py generate
```

Example output:

```text
He eagerly walked until he kept jogging again.
```

Output varies each time because the next word is sampled randomly from the model's learned probabilities.

## Project structure

```text
main.py                 Train the model or generate text
download_corpus.py      Download and save the TinyStories corpus
config.py               Corpus and saved-model paths
tokenizer.py            Convert text into tokens
sentenceprocessor.py    Add sentence boundary markers
trainer.py              Count word-pair occurrences
bigram.py               Convert counts into probabilities
serializer.py           Save and load the model as JSON
generator.py            Sample words to generate a sentence
```

## Limitations

A bigram model chooses the next word using only the immediately preceding word. It cannot maintain long-range context, reliably follow grammar, or understand meaning across a sentence. As a result, some generated sentences will be awkward or incoherent.

Increasing the corpus size can improve common word transitions, but it does not remove the fundamental bigram limitation. Useful next experiments include a larger TinyStories corpus, trigram models, and neural language models.

## Data and generated files

The corpus in `dataset/`, saved models in `models/`, and the `.venv/` environment are intentionally ignored by Git. They can be recreated using the commands above and should not be committed to the repository.
