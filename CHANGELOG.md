# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-09-27

### Added
- **Core Bigram Language Model**:
  - Maximum Likelihood Estimation (MLE) computing bigram conditional probabilities ($P(w_i \mid w_{i-1})$).
  - Punctuation-aware tokenizer isolating ASCII and Unicode typographic punctuation (`“`, `”`, `‘`, `’`, `—`, `–`, `…`).
  - Sentence boundary handling injecting `<START>` and `<END>` boundary tokens.
  - Autoregressive text generation with inverse transform / roulette-wheel probabilistic sampling.
  - Temperature scaling ($P_T \propto P^{1/T}$) supporting greedy decoding ($T \le 0.05$), balanced sampling ($T \approx 0.7$), and creative high-entropy sampling ($T \ge 1.5$).
  - Punctuation-aware detokenizer formatting spaces around punctuation, quotes, contractions, and brackets.
  - Intrinsic evaluation suite computing vocabulary size, unique bigram counts, test set bigram coverage, log-likelihood, and perplexity.
  - Model serialization and loading to/from validated JSON schema.
  - Streaming dataset downloader for TinyStories (Hugging Face) with deterministic 80/20 train/test SHA-256 story hash split.
- **Interactive Google Colab Demo**:
  - Self-contained interactive Jupyter notebook (`notebooks/demo.ipynb`) with an embedded fairy tales mini-corpus for zero-download instant training.
  - Step-by-step walkthrough covering tokenization, boundary injection, transition matrix inspection, temperature slider, and intrinsic perplexity evaluation.
- **Unified Command-Line Interface (CLI)**:
  - PEP 621 package metadata in `pyproject.toml` with `simple-bigram` console script entry point.
  - Subcommands: `train`, `generate`, and `evaluate` with customizable CLI flags (`--temperature`, `--max-tokens`, `--seed`, etc.).
- **Comprehensive Automated Test Suite**:
  - 67 unit tests covering 100% of core pipeline components with 98% overall line coverage.
  - Edge case coverage: empty strings, isolated punctuation, consecutive sentence terminators, floating-point precision bounds, and perplexity overflow protection.
- **Documentation & Open-Source Governance**:
  - Production-grade `README.md` with pipeline Mermaid flowchart, architectural breakdown, and quickstart commands.
  - Open-source governance files: `LICENSE` (MIT), `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1), `CONTRIBUTING.md`, and `SECURITY.md`.

### Fixed
- Fixed leading whitespace bug in detokenizer where initial sentence tokens received an unwanted leading space.
- Fixed cumulative sampling loop termination failure caused by floating-point precision round-off.
- Fixed consecutive sentence terminator handling (`...`, `?!`) preventing spurious empty sentences.
- Fixed potential math overflow in perplexity calculation when exponentiating large negative log-probabilities.

[Unreleased]: https://github.com/thisiskamaljoshi/simple-bigram-model/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/thisiskamaljoshi/simple-bigram-model/releases/tag/v1.0.0
