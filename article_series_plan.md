# Technical Article Series Plan: Statistical Bigram Language Model from Scratch

A comprehensive roadmap for a 5-part technical article series documenting the theory, design decisions, and from-scratch Python implementation of a Statistical Bigram Language Model.

---

## Series Overview

Language models are fundamentally machines that assign probabilities to sequences of words. While modern AI focuses on multi-billion parameter Transformers, mastering classical statistical language models (n-grams, Markov chains, Maximum Likelihood Estimation) builds an intuitive foundation for understanding autoregressive generation, vocabulary management, tokenization, and evaluation metrics.

This 5-part series mirrors the end-to-end NLP engineering lifecycle:
$$\text{Data Ingestion} \longrightarrow \text{Preprocessing} \longrightarrow \text{Model Training} \longrightarrow \text{Text Generation} \longrightarrow \text{Intrinsic Evaluation}$$

---

## Article Breakdown

### Article 1: Building a Streaming Data Pipeline with Deterministic Train/Test Hashing

- **Summary**: Explains how to efficiently ingest large language corpora from Hugging Face without loading entire datasets into RAM by streaming text up to a byte limit. It explores how to implement deterministic, leakage-free 80/20 train/test splits at the story level using SHA-256 cryptographic hashing, and how to use atomic file writes (`.tmp` to final destination) to prevent corrupt files.
- **Repository Files Discussed**:
  - [`download_corpus.py`](download_corpus.py) — `CorpusDownloader`, streaming logic, SHA-256 story bucket hashing, atomic temp-file swapping.
  - [`config.py`](config.py) — Target sizes, corpus definitions, and split path configurations.
  - [`corpus.py`](corpus.py) — File loading utilities and UTF-8 validation.
- **Key Concepts Learned**:
  - Memory-efficient data ingestion via `datasets.load_dataset(..., streaming=True)`.
  - Cryptographic story-level hashing (SHA-256 bucket allocation) for reproducible, non-random train/test splits.
  - Safe atomic file operations (`os.replace`) to prevent corrupted datasets.
- **Why It Deserves a Separate Article**: In modern machine learning, data engineering is an essential prerequisite. Combining data streaming and cryptographic splitting with text tokenization would overload the reader and distract from NLP-specific concepts.

---

### Article 2: Custom Tokenization and the Art of Sentence Boundary Modeling

- **Summary**: Dives into why basic whitespace splitting fails on natural language text and how to build a character-level tokenization engine that cleanly isolates punctuation marks. It then explores the theoretical and practical necessity of artificial boundary tokens (`<START>` and `<END>`) to model how sentences begin and conclude.
- **Repository Files Discussed**:
  - [`tokenizer.py`](tokenizer.py) — Character-level buffer scanner and punctuation isolation.
  - [`sentenceprocessor.py`](sentenceprocessor.py) — Sentence boundary injection (`<START>` / `<END>`).
  - [`helpers.py`](helpers.py) — Punctuation sets and character constants.
- **Key Concepts Learned**:
  - The limitations of whitespace splitting and why punctuation marks must be treated as independent tokens.
  - The mathematics of boundary modeling: modeling $P(w_1 \mid \langle\text{START}\rangle)$ for initial words and $P(\langle\text{END}\rangle \mid w_n)$ for sentence terminations.
  - Transforming continuous token streams into structured sentence sequences.
- **Why It Deserves a Separate Article**: Tokenization and sequence boundary formatting are the fundamental bridges between unstructured text and statistical learning. Bundling this with the mathematical training phase would obscure the nuances of text normalization and token tagging.

---

### Article 3: Maximum Likelihood Estimation: Learning and Serializing Bigram Transition Distributions

- **Summary**: Details the core learning algorithm: scanning adjacent token pairs across the corpus to build frequency counts and converting them into conditional probability distributions $P(w_i \mid w_{i-1})$ using Maximum Likelihood Estimation (MLE). It also details how to validate that probability axioms hold ($\sum P = 1.0$) and how to serialize sparse transition matrices to validated JSON.
- **Repository Files Discussed**:
  - [`trainer.py`](trainer.py) — Scanning adjacent token pairs and building frequency tables.
  - [`bigram.py`](bigram.py) — Maximum Likelihood Estimation normalization.
  - [`serializer.py`](serializer.py) — JSON serialization and schema validation.
- **Key Concepts Learned**:
  - The first-order Markov assumption: approximating full document history using only the immediate predecessor:
    $$P(w_1, w_2, \dots, w_N) \approx \prod_{i=1}^{N} P(w_i \mid w_{i-1})$$
  - Maximum Likelihood Estimation (MLE) formula for bigrams:
    $$P(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\sum_{w'} \text{Count}(w_{i-1}, w')}$$
  - Probability axiom verification and nested dictionary serialization.
- **Why It Deserves a Separate Article**: This is the mathematical core of the model. It focuses strictly on how statistical counting translates into conditional probabilities, providing a dedicated space for the math and data structure design.

---

### Article 4: Autoregressive Text Generation: Probabilistic Sampling and Detokenization

- **Summary**: Demonstrates how to take a trained bigram probability matrix and autoregressively synthesize new sentences token by token starting from `<START>` until an `<END>` token is sampled. It explains inverse transform (roulette-wheel) cumulative sampling and details how to reconstruct clean, human-readable text from raw tokens using punctuation attachment rules.
- **Repository Files Discussed**:
  - [`sampler.py`](sampler.py) — Probabilistic sampling over cumulative distributions.
  - [`generator.py`](generator.py) — Autoregressive generation loop from `<START>` to `<END>`.
  - [`detokenizer.py`](detokenizer.py) — Punctuation attachment rules (`NO_SPACE_BEFORE`, `NO_SPACE_AFTER`, `NO_SPACE_AROUND`).
- **Key Concepts Learned**:
  - The autoregressive loop in language generation (output of step $t$ becomes conditioning input for step $t+1$).
  - Probabilistic sampling via cumulative probability distributions (inverse transform sampling) vs. greedy argmax.
  - Rule-based detokenization: handling whitespace rules before, after, and around punctuation marks (parentheses, quotes, periods).
- **Why It Deserves a Separate Article**: Text generation (inference) and detokenization are distinct from training and evaluation. It answers the question: *"Now that we have probabilities, how do we turn them into fluent natural language outputs?"*

---

### Article 5: Evaluating Language Models: Test Coverage, Perplexity, and the Reality of the Zero-Frequency Problem

- **Summary**: Explores intrinsic evaluation of language models on unseen held-out data by measuring vocabulary size, learned bigrams, and test bigram coverage ($97.89\%$). It then breaks down the mathematical calculation of log-likelihood and perplexity, analyzing the real-world result encountered in unsmoothed MLE models: why even a high coverage rate still results in $-\infty$ log-probability and $\infty$ perplexity due to the Zero-Frequency Problem.
- **Repository Files Discussed**:
  - [`evaluator.py`](evaluator.py) — Axiom validation, bigram coverage, log-probability, and perplexity computation.
  - [`main.py`](main.py) — CLI evaluation orchestration and reporting.
- **Key Concepts Learned**:
  - Measuring n-gram coverage on held-out test splits (seen vs. unseen bigrams).
  - The mathematics of average log-probability and perplexity:
    $$\text{Log-Probability} = \sum_{i=1}^{N} \ln P(w_i \mid w_{i-1})$$
    $$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \ln P(w_i \mid w_{i-1})\right)$$
  - The Zero-Frequency Problem: why unsmoothed MLE assigns $P(w_i \mid w_{i-1}) = 0$ to unseen transitions, causing mathematical divergence ($\text{Perplexity} = \infty$), and why this is a fundamental milestone in statistical NLP.
- **Why It Deserves a Separate Article**: Evaluation provides the critical theoretical insight of the entire project. Discussing the exact reasons why unsmoothed perplexity results in $\infty$ on real-world test data is a high-value lesson for learners transitioning into statistical language modeling.

---

## Recommended Publishing Order & Summary

| # | Article Title | Focus Area | Key Code Files |
| :-: | :--- | :--- | :--- |
| **1** | **Building a Streaming Data Pipeline with Deterministic Train/Test Hashing** | Data Ingestion & Splitting | [`download_corpus.py`](download_corpus.py), [`config.py`](config.py), [`corpus.py`](corpus.py) |
| **2** | **Custom Tokenization and the Art of Sentence Boundary Modeling** | Text Preprocessing & Tags | [`tokenizer.py`](tokenizer.py), [`sentenceprocessor.py`](sentenceprocessor.py), [`helpers.py`](helpers.py) |
| **3** | **Maximum Likelihood Estimation: Learning and Serializing Bigram Transition Distributions** | MLE Training & Serialization | [`trainer.py`](trainer.py), [`bigram.py`](bigram.py), [`serializer.py`](serializer.py) |
| **4** | **Autoregressive Text Generation: Probabilistic Sampling and Detokenization** | Sampling & Detokenization | [`sampler.py`](sampler.py), [`generator.py`](generator.py), [`detokenizer.py`](detokenizer.py) |
| **5** | **Evaluating Language Models: Test Coverage, Perplexity, and the Reality of the Zero-Frequency Problem** | Intrinsic Evaluation & Theory | [`evaluator.py`](evaluator.py), [`main.py`](main.py) |

---

## Why 5 Articles is the Ideal Scope

1. **Neither Too Many Nor Too Few**: 
   - 3 or 4 articles would force distinct engineering domains together (e.g., merging Data Streaming with Tokenization, or combining Autoregressive Generation with Test Perplexity Evaluation).
   - 6 or more articles would inflate minor utility files (`helpers.py`, `serializer.py`) into superficial, low-value articles.
2. **Follows Natural AI Engineering Workflow**: A reader can follow along from raw internet data to text preprocessing, mathematical modeling, text synthesis, and rigorous metric evaluation.
3. **Showcases Real-World Findings**: Concludes with a mathematically honest exploration of the Zero-Frequency problem, turning a potential point of confusion into a deep learning milestone.
