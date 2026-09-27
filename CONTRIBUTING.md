# Contributing to Statistical Bigram Language Model

Thank you for your interest in contributing to **Statistical Bigram Language Model**! We welcome contributions, bug reports, feature requests, and improvements.

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## How to Contribute

### 1. Reporting Bugs

If you find a bug:
1. Check the [existing issues](https://github.com/thisiskamaljoshi/simple-bigram-model/issues) to ensure it hasn't already been reported.
2. Open a new issue using the **Bug Report** template.
3. Include clear steps to reproduce the issue, along with your OS and Python version.

### 2. Suggesting Features

We welcome ideas for improving the project (such as adding smoothing techniques, higher-order n-grams, or performance improvements):
1. Open an issue using the **Feature Request** template.
2. Explain the motivation, proposed solution, and any alternatives considered.

### 3. Pull Requests

1. **Fork** the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Set up the virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. **Make your changes**:
   - Write clear, idiomatic Python adhering to PEP 8.
   - Include type hints for function signatures.
   - Add docstrings explaining mathematical logic where applicable.
5. **Run tests**:
   Ensure all existing and new tests pass, and coverage remains above 90%:
   ```bash
   # Run tests
   pytest

   # Run tests with terminal coverage report
   pytest --cov=. --cov-report=term-missing tests/
   ```
6. **Submit your Pull Request**:
   - Fill out the PR template with details of what was changed and why.
   - Reference any relevant issue numbers (e.g., `Fixes #12`).

---

## Project Structure

```text
simple-bigram-model/
├── dataset/             # Corpus data directory (gitignored)
├── docs/                # Architecture docs, talk plans, and article roadmaps
├── models/              # Saved model JSON artifacts (gitignored)
├── tests/               # Automated test suite
├── bigram.py            # MLE transition probability calculation
├── config.py            # Central configuration and corpus paths
├── corpus.py            # Text corpus loader utility
├── detokenizer.py       # Punctuation-aware detokenizer
├── download_corpus.py   # Streaming dataset downloader with SHA-256 splits
├── evaluator.py         # Model evaluation suite (coverage, perplexity)
├── generator.py         # Autoregressive text generator
├── helpers.py           # Punctuation constants and formatting sets
├── main.py              # CLI entry point (train, generate, evaluate)
├── sampler.py           # Probabilistic roulette-wheel sampler
├── sentenceprocessor.py # Boundary tokens (<START> / <END>)
├── serializer.py        # Model JSON serializer and schema validator
└── tokenizer.py         # Character-level punctuation-isolating tokenizer
```

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
