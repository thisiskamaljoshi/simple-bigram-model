# Statistical Bigram Language Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thisiskamaljoshi/simple-bigram-model/blob/main/notebooks/demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A modular statistical language model built from scratch in Python. It learns word transition probabilities from the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, generates text by probabilistic sampling, and provides an evaluation suite to measure test coverage, log-likelihood, and perplexity.

This project demonstrates core concepts in Natural Language Processing (NLP) and statistical language modeling: text preprocessing, tokenization, vocabulary construction, Maximum Likelihood Estimation (MLE), autoregressive sampling, punctuation-aware detokenization, and intrinsic model evaluation.

---

## Pipeline Overview

```mermaid
flowchart TD
    subgraph Data_Preparation ["Data Preparation"]
        HF["TinyStories Dataset"] -->|Stream & SHA256 Split| DC["download_corpus.py"]
        DC --> TrainTxt["Train Corpus: 80%"]
        DC --> TestTxt["Test Corpus: 20%"]
    end

    subgraph Training_Pipeline ["Training Pipeline"]
        TrainTxt --> CorpusLoad["corpus.py"]
        CorpusLoad --> Tokenize["tokenizer.py"]
        Tokenize --> SentProc["sentenceprocessor.py<br/>Add START / END"]
        SentProc --> Trainer["trainer.py<br/>Count Bigram Pairs"]
        Trainer --> Bigram["bigram.py<br/>Calculate P(w_i | w_i-1)"]
        Bigram --> Serializer["serializer.py<br/>Save to JSON"]
        Serializer --> ModelFile[("models/tinystories_50mb_bigrams.json")]
    end

    subgraph Generation_Pipeline ["Generation Pipeline"]
        ModelFile --> LoadGen["serializer.py"]
        LoadGen --> Sampler["sampler.py<br/>Probabilistic Sampling"]
        Sampler --> Generator["generator.py<br/>Generate Tokens"]
        Generator --> Detokenizer["detokenizer.py<br/>Format Spacing"]
        Detokenizer --> GenText["Generated Sentence"]
    end

    subgraph Evaluation_Pipeline ["Evaluation Pipeline"]
        ModelFile --> LoadEval["serializer.py"]
        TestTxt --> TestLoad["corpus.py"]
        TestLoad --> TestTok["tokenizer.py & sentenceprocessor.py"]
        LoadEval --> Evaluator["evaluator.py"]
        TestTok --> Evaluator
        Evaluator --> Metrics["Coverage, Log-Prob, Perplexity Report"]
    end
```

---

## Features

- **Streaming Dataset Downloader**: Streams TinyStories and creates a deterministic 80/20 train/test split using SHA-256 story hashing.
- **Custom Tokenization**: Splits words while isolating punctuation marks.
- **Sentence Boundary Handling**: Automatically wraps sentences with `<START>` and `<END>` tokens for natural beginnings and terminations.
- **Maximum Likelihood Estimation (MLE)**: Computes bigram conditional transition probabilities:
  $$P(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\sum_{w'} \text{Count}(w_{i-1}, w')}$$
- **Model Serialization & Validation**: Saves trained probability distributions as compact JSON, with schema and probability distribution validation.
- **Probabilistic Sampling**: Employs inverse transform (roulette wheel) sampling to generate diverse text based on learned distributions.
- **Punctuation-Aware Detokenization**: Reconstructs natural sentences with custom rules for spaces before, after, and around punctuation marks.
- **Evaluation Suite**:
  - Validates probability distributions ($\sum P = 1.0$).
  - Measures vocabulary size and unique learned bigrams.
  - Computes test set bigram coverage (seen vs. unseen bigrams).
  - Calculates total and average log-probabilities and perplexity.

---

## Setup

### Prerequisites
- Python 3.11 or newer
- Internet connection (for downloading dependencies and dataset)

### Installation

```powershell
git clone <your-repository-url>
cd simple-bigram-model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To install development and testing dependencies (`pytest`, `pytest-cov`):

```powershell
python -m pip install -r requirements-dev.txt
```

To install the project as an editable CLI tool (enables the `simple-bigram` command):

```powershell
python -m pip install -e .
```

If your network uses a custom Windows certificate and `pip` reports an SSL verification error:

```powershell
python -m pip install --use-feature=truststore -r requirements.txt
```

---

## Usage

The project can be explored interactively in Google Colab or via the unified CLI (`simple-bigram` or `python main.py`).

### 0. Interactive Demo (Google Colab)

Want to try the model immediately without any local setup or file downloads? Run our interactive walkthrough notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thisiskamaljoshi/simple-bigram-model/blob/main/notebooks/demo.ipynb)

The demo notebook [`notebooks/demo.ipynb`](notebooks/demo.ipynb) includes:
- **Embedded Fairy Tales Mini-Corpus**: Zero-download training in milliseconds.
- **Tokenization & Sentence Boundaries**: Visualizing punctuation isolation and `<START>` / `<END>` injection.
- **Transition Matrix Inspection**: Examining conditional counts, probability distributions, and 2D heatmaps.
- **Temperature Scaling Comparison**: Interactive slider comparing Greedy ($T=0.01$), Balanced ($T=0.7$), and Creative ($T=1.5$) generation.
- **Intrinsic Evaluation**: Calculating test set coverage, log-likelihood, and perplexity.

### 1. Download Corpus

Downloads the configured 50 MB TinyStories corpus and creates the 80% train and 20% test splits in `dataset/tinystories/`:

```powershell
python download_corpus.py
```

### 2. Train the Model
Reads the training split, counts bigram transitions, computes conditional probabilities, and saves the model to `models/tinystories_50mb_bigrams.json`:

```powershell
python main.py train
```

### 3. Generate Text
Loads the saved model and generates a sentence starting from `<START>` until `<END>` or the token limit:

```powershell
python main.py generate
```

#### Optional Generation Flags:
- `--temperature`: Sampling temperature. Lower values (e.g. `0.2` or `0.0001`) produce greedy/deterministic text; higher values (e.g. `1.2`) produce more random text. Default is `1.0`.
- `--max-tokens`: Maximum number of tokens to generate. Default is `100`.
- `--seed`: Integer random seed for reproducible outputs.

```powershell
# Greedy / deterministic generation
python main.py generate --temperature 0.001

# Reproducible sampling with seed and max length
python main.py generate --temperature 0.8 --max-tokens 50 --seed 42
```

*Example Output:*
```text
One day was selfish.
```

### 4. Evaluate the Model
Evaluates the trained model against the held-out test corpus:

```powershell
python main.py evaluate
```

*Example Output:*
```text
===== Bigram Model Evaluation =====
Vocabulary size: 17369
Unique learned bigrams: 436640
Evaluated bigrams: 2910351
Seen bigrams: 2849001
Unseen bigrams: 61350
Coverage: 97.89%
Total log probability: -inf
Average log probability: -inf
Perplexity: inf
```

---

## Testing & Code Coverage

The project includes an automated unit test suite (70 tests across 14 test modules) verifying all components, edge cases, and numerical properties, achieving **97.3% test coverage**.

### 1. Install Testing Dependencies
If not already installed, install the developer dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

### 2. Run the Test Suite
Run all tests using `pytest`:

```powershell
python -m pytest
```

For verbose output with individual test names and status:

```powershell
python -m pytest tests/ -v
```

### 3. Check Test Coverage in Terminal
Display test coverage with missing line indicators in the terminal:

```powershell
python -m pytest --cov=. --cov-report=term-missing tests/
```

### 4. Generate Interactive HTML Coverage Report
Generate an interactive HTML report to visually explore covered and uncovered lines across all source files:

```powershell
python -m pytest --cov=. --cov-report=html tests/
```

Once generated, open `htmlcov/index.html` in your web browser.

---

## Evaluation Metrics Explained

- **Vocabulary Size**: Number of unique tokens across the training distribution.
- **Unique Learned Bigrams**: Number of distinct $(w_{i-1}, w_i)$ pairs observed during training.
- **Evaluated Bigrams**: Total bigram count in the test corpus.
- **Seen vs. Unseen Bigrams**: Counts how many test bigrams were learned during training vs. completely new word transitions.
- **Coverage**: The percentage of test bigrams present in the trained model ($\frac{\text{Seen Bigrams}}{\text{Evaluated Bigrams}}$).
- **Log Probability & Perplexity**:
  $$\text{Log Probability} = \sum_{i=1}^{N} \ln P(w_i \mid w_{i-1})$$
  $$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \ln P(w_i \mid w_{i-1})\right)$$
  > [!NOTE]
  > Because raw MLE bigram models do not apply smoothing (such as Laplace or Kneser-Ney), any unseen test bigram has $P(w_i \mid w_{i-1}) = 0$. This correctly results in a log-probability of $-\infty$ and perplexity of $\infty$, highlighting the zero-frequency problem in unsmoothed n-gram models.

---

## Project Structure

| File / Directory | Description |
| :--- | :--- |
| `notebooks/demo.ipynb` | Interactive walkthrough & Colab demo with embedded fairy tales mini-corpus |
| `CHANGELOG.md` | Version history and release notes following Keep a Changelog standard |
| `SECURITY.md` | Security policy, supported versions, and vulnerability reporting process |
| `docs/` | Comprehensive technical articles, talk plans, and refactoring roadmap |
| `tests/` | Automated unit test suite run with `pytest` |
| `main.py` | CLI entry point supporting `train`, `generate`, and `evaluate` commands |
| `config.py` | Dataset definitions, file paths, and target split sizes |
| `download_corpus.py` | Streams TinyStories and creates deterministic 80/20 train/test splits |
| `corpus.py` | File loader utility with validation and UTF-8 handling |
| `tokenizer.py` | Splits text into words while isolating punctuation marks |
| `sentenceprocessor.py` | Adds sentence boundary tokens (`<START>` and `<END>`) |
| `trainer.py` | Counts adjacent token pair frequencies across the corpus |
| `bigram.py` | Computes conditional probabilities $P(w_i \mid w_{i-1})$ |
| `serializer.py` | Serializes and loads probability maps to/from validated JSON |
| `sampler.py` | Probabilistic sampling using cumulative distributions |
| `generator.py` | Autoregressively generates sentences using the learned model |
| `detokenizer.py` | Reassembles tokens into natural, properly spaced text |
| `helpers.py` | Punctuation sets and spacing attachment rules |
| `evaluator.py` | Validates probability axioms and calculates coverage, log-prob, and perplexity |

---

## Limitations and Future Experiments

1. **Context Window (First-Order Markov Property)**: Bigram models predict the next word using *only* the single preceding word ($n=2$). They cannot maintain long-term context or complex grammatical agreement.
2. **Zero-Frequency Problem**: Unseen word transitions have zero probability. Implementing smoothing techniques (e.g., Laplace Add-1, Good-Turing, or Kneser-Ney smoothing) will enable finite perplexity scores on unseen text.
3. **Future Experiments**:
   - **Higher-order N-grams**: Trigram ($n=3$) and 4-gram models with backoff or interpolation.
   - **Smoothing Techniques**: Additive smoothing and backoff to unigram probabilities.
   - **Subword Tokenization**: Byte-Pair Encoding (BPE) or WordPiece tokenizers.
   - **Neural Language Models**: Transitioning to MLP / RNN / Transformer architectures.

---

## Data and Generated Files

The corpus split files in `dataset/`, serialized model files in `models/`, and `.venv/` virtual environment are excluded by `.gitignore`. They can be recreated at any time using the commands documented above.

---

## Contributing

Contributions, issues, and feature requests are welcome! Please check out the [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Security

Please review our [Security Policy](SECURITY.md) for supported versions and vulnerability reporting guidelines.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full record of changes, architectural additions, bug fixes, and release notes.

---

## License

This project is open-source and licensed under the [MIT License](LICENSE).


