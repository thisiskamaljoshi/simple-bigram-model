# Technical Talks Plan: Statistical Bigram Language Model from Scratch

A comprehensive guide and outline for delivering technical talks, meetup sessions, conference presentations, and engineering brown bags based strictly on this from-scratch Python language model implementation.

---

## Overview of the Talk Portfolio

This portfolio is designed to give you options for different speaking scenarios:
- **1 Flagship Masterclass Talk (45–60 min)**: Covers the full end-to-end architecture and live generation. Perfect for conference sessions, meetup headliners, or university guest lectures.
- **4 Specialized Track Talks (25–30 min each)**: Zeroes in on specific engineering domains (Data Engineering, NLP Preprocessing, Generative Sampling, or Metric Mathematics).

---

## Talk 1: Language Models from First Principles: Building a Statistical Bigram Model in Pure Python

- **Format & Duration**: 45-minute Flagship Talk / 60-minute Interactive Workshop
- **Ideal Audience**: General Software Engineers, AI/ML Learners, Data Scientists, Tech Meetup Attendees
- **Narrative Hook**: *"Before Transformers, attention heads, and GPUs, how does language generation actually work from first principles? Let's build an autoregressive language model in pure Python."*
- **Description**: Walks through the full lifecycle of building a statistical language model from scratch. Starting from raw text streaming on TinyStories, the talk explains custom tokenization, Maximum Likelihood Estimation of bigram distributions, autoregressive sampling, and intrinsic model evaluation.
- **Repository Files to Showcase**:
  - [`download_corpus.py`](download_corpus.py) — Ingestion & deterministic splitting.
  - [`tokenizer.py`](tokenizer.py) & [`sentenceprocessor.py`](sentenceprocessor.py) — Sequence prep & boundary tags.
  - [`bigram.py`](bigram.py) & [`trainer.py`](trainer.py) — Counting & MLE transition probabilities.
  - [`generator.py`](generator.py) & [`detokenizer.py`](detokenizer.py) — Autoregressive sampling & text reconstruction.
  - [`evaluator.py`](evaluator.py) — Model evaluation & perplexity.
- **Key Concepts Learned**:
  - The first-order Markov assumption: $P(w_1, \dots, w_N) \approx \prod P(w_i \mid w_{i-1})$.
  - End-to-end NLP engineering pipeline.
  - Why bigrams excel at local phrasing but lack global long-term coherence.
- **Live Demo Script**:
  1. Open terminal and run `python main.py generate` multiple times to show stochastic generation.
  2. Inspect the saved JSON model in `models/tinystories_50mb_bigrams.json` to show the conditional probability map.
  3. Run `python main.py evaluate` to display the evaluation metrics live.

---

## Talk 2: Data Pipelines for NLP: Streaming Massive Datasets and Deterministic SHA-256 Splitting

- **Format & Duration**: 25-minute Deep Dive
- **Ideal Audience**: Data Engineers, ML Platform Engineers, Backend Developers
- **Narrative Hook**: *"How do you reliably ingest web-scale text on a standard laptop without running out of RAM or introducing subtle train/test leakage?"*
- **Description**: Focuses strictly on the data engineering challenges of NLP. Explains how to stream large datasets sample-by-sample up to a fixed byte budget, apply document-level cryptographic SHA-256 hashing for leak-free, deterministic 80/20 train/test splits, and use atomic file swapping to guarantee zero data corruption.
- **Repository Files to Showcase**:
  - [`download_corpus.py`](download_corpus.py) — `CorpusDownloader`, streaming generator, hash bucketing, atomic temp files.
  - [`config.py`](config.py) — Corpus configurations and dataset splits.
- **Key Concepts Learned**:
  - Streaming iterators via `datasets.load_dataset(..., streaming=True)` vs. in-memory loading.
  - Cryptographic story hashing (`int.from_bytes(hashlib.sha256(text).digest()[:8]) % 100`) for reproducible splits.
  - POSIX-compliant atomic file write patterns (`os.replace`) to prevent partially written corpus files.
- **Live Demo Script**:
  1. Show a small Python script calculating SHA-256 buckets for different sample stories.
  2. Run `python download_corpus.py` and observe the streaming progress and deterministic 80/20 split summary.

---

## Talk 3: Beyond `text.split()`: Punctuation-Aware Tokenization and Sentence Boundary Modeling

- **Format & Duration**: 25-minute Deep Dive
- **Ideal Audience**: Python Developers, Junior NLP Engineers, CS Students
- **Narrative Hook**: *"Why is text.split(' ') the single biggest trap in NLP, and how do language models know when a sentence is over?"*
- **Description**: Dives into the mechanics of converting unstructured text into structured tokens. Demonstrates why basic whitespace splitting fails on punctuation and contractions, how to build a character-level tokenization engine, and why injecting artificial `<START>` and `<END>` boundary tokens is mathematically mandatory for learning initial and terminal word probabilities.
- **Repository Files to Showcase**:
  - [`tokenizer.py`](tokenizer.py) — Buffer-based word splitting and punctuation isolation.
  - [`sentenceprocessor.py`](sentenceprocessor.py) — Injection of `<START>` and `<END>` boundary tokens.
  - [`helpers.py`](helpers.py) — Punctuation sets (`NO_SPACE_BEFORE`, `NO_SPACE_AFTER`, `NO_SPACE_AROUND`).
- **Key Concepts Learned**:
  - Isolating punctuation marks as independent vocabulary tokens.
  - Modeling sentence start distributions $P(w_1 \mid \langle\text{START}\rangle)$ and stop distributions $P(\langle\text{END}\rangle \mid w_n)$.
  - Turning streaming text tokens into clean sentence structures.
- **Live Demo Script**:
  1. Open a Python REPL and run the custom tokenizer on a complex sentence containing parentheses, quotes, and punctuation.
  2. Pass the tokens into `SentenceProcessor` and display the resulting `<START>` / `<END>` sequence in the terminal.

---

## Talk 4: The Mechanics of Autoregressive Sampling: Roulette-Wheel Distributions and Detokenization

- **Format & Duration**: 30-minute Deep Dive
- **Ideal Audience**: AI Engineers, Algorithm Enthusiasts, Python Developers
- **Narrative Hook**: *"Once you have a probability distribution, how do you turn math into human-like text without getting stuck in infinite loops?"*
- **Description**: Details the autoregressive generation loop. Contrasts greedy argmax decoding with cumulative distribution (roulette-wheel) inverse transform sampling, and explains how to format raw token outputs into clean sentences using punctuation attachment matrices.
- **Repository Files to Showcase**:
  - [`sampler.py`](sampler.py) — Inverse transform cumulative probability sampling.
  - [`generator.py`](generator.py) — Autoregressive generation loop from `<START>` to `<END>`.
  - [`detokenizer.py`](detokenizer.py) — Punctuation spacing attachment rules.
- **Key Concepts Learned**:
  - The autoregressive feedback loop: $w_t \sim P(\cdot \mid w_{t-1})$.
  - Cumulative distribution sampling (roulette-wheel selection).
  - Punctuation attachment rules for natural reading flow.
- **Live Demo Script**:
  1. Step through the `Generator` step-by-step in a debugger or REPL.
  2. At each step, print the candidate probability distribution for the current word.
  3. Show the raw token list `["Once", "upon", "a", "time", ",", "there", "was"]` transforming into formatted text `"Once upon a time, there was"`.

---

## Talk 5: Why My Model Hit Infinite Perplexity: Intrinsic Evaluation and the Zero-Frequency Problem

- **Format & Duration**: 30-minute Deep Dive
- **Ideal Audience**: Machine Learning Engineers, Data Science Students, Math for ML Enthusiasts
- **Narrative Hook**: *"My language model achieved 97.89% coverage on test data, but the perplexity was infinite. Here is why that is actually mathematically correct."*
- **Description**: Unpacks intrinsic evaluation metrics for language models: vocabulary size, transition coverage, log-likelihood, and perplexity. Explains the mathematics of cross-entropy and reveals the brutal reality of the Zero-Frequency Problem in unsmoothed Maximum Likelihood Estimation.
- **Repository Files to Showcase**:
  - [`evaluator.py`](evaluator.py) — Axiom validation, bigram coverage, log-probability, and perplexity computation.
  - [`main.py`](main.py) — CLI evaluation orchestration and reporting.
- **Key Concepts Learned**:
  - Validating probability distribution axioms ($\sum P(w \mid \text{word}) = 1.0$).
  - Computing test set bigram coverage ($\frac{\text{Seen Bigrams}}{\text{Total Bigrams}}$).
  - The mathematics of Perplexity:
    $$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \ln P(w_i \mid w_{i-1})\right)$$
  - The Zero-Frequency Problem: why an unseen bigram with $P=0$ drives log-probability to $-\infty$ and perplexity to $\infty$.
- **Live Demo Script**:
  1. Run `python main.py evaluate` and examine the terminal output.
  2. Find an unseen bigram in the test dataset that was never seen in training (e.g., a specific name or rare word pairing) and demonstrate why $\ln(0) = -\infty$.

---

## Talk Delivery Summary

| Talk # | Title | Format / Duration | Best Fit |
| :---: | :--- | :---: | :--- |
| **1** | **Language Models from First Principles** | 45–60 min Keynote / Workshop | Tech conferences, main meetup tracks, university lectures |
| **2** | **Data Pipelines for NLP** | 25 min Deep Dive | Data engineering meetups, ML infrastructure sessions |
| **3** | **Beyond `text.split()`** | 25 min Deep Dive | Python user groups, introductory NLP workshops |
| **4** | **The Mechanics of Autoregressive Sampling** | 30 min Deep Dive | AI engineering meetups, algorithm & systems tracks |
| **5** | **Why My Model Hit Infinite Perplexity** | 30 min Deep Dive | Data science meetups, math & theory study groups |
