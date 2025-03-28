# Memorization Detection Implementation

This directory contains the implementation of four memorization detection methods specifically adapted for the MedQA-USMLE dataset, which is a multiple-choice medical exam benchmark.

## Structure

- `config.py` - Configuration settings
- `utils/` - Utility functions
  - `api.py` - OpenRouter API client
  - `data_loader.py` - Functions for loading and processing MedQA dataset
- `detection_methods/` - Implementation of detection methods
  - `perplexity.py` - Perplexity-based method
  - `ngram_overlap.py` - N-gram overlap method
  - `embedding_similarity.py` - Embedding similarity method
  - `consistency_testing.py` - Consistency testing method

## Detection Methods

### 1. Perplexity-Based Method

Measures how surprised a model is by each answer option. If a model assigns unusually low perplexity (high probability) to the correct answer compared to incorrect answers, this may indicate memorization.

### 2. N-gram Overlap Method

Analyzes overlap between n-grams in the model's explanation and those in the question/options. High overlap may indicate verbatim copying from training data.

### 3. Embedding Similarity Method

Computes semantic similarity between model outputs and reference texts using embeddings. Unusually high similarity may suggest memorization.

### 4. Consistency Testing Method

Tests if the model's answer remains consistent when the question or options are slightly modified. Models that have memorized specific question-answer pairs may show inconsistent behavior with modifications.

## Usage

See the main README and `main.py` for usage instructions.
