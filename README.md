# LLM Memorization Detection Tool

This repository contains tools for measuring whether Large Language Models (LLMs) have memorized their training data, applicable to both open-source and closed-source models.

## Overview

Memorization in LLMs refers to the phenomenon where models directly reproduce content from their training data instead of generating novel responses. This tool provides multiple methods to detect such memorization, which is crucial for:

- Ensuring data privacy and preventing data leakage
- Evaluating model generalization abilities
- Understanding model behavior and limitations
- Identifying potential copyright infringement risks

## Critical Factors Affecting Memorization

Research reveals several key factors that influence memorization rates in LLMs:

- **Model Size**: Larger models memorize faster and retain more data (125M vs. 1.3B parameter models show 4× memorization rate gap) [1][6]
- **Data Duplication**: Repeated examples in training data increase memorization risk (log-linear relationship) [6]
- **Context Length**: Longer generation windows expose more memorized content [3][6]
- **Dataset Entropy**: Unique or rare information is more easily identified when memorized [4][5]

## Methods for Detecting Memorization

### 1. Perplexity-Based Detection

**Description**: Measures how surprised a model is by a given text sequence. Lower perplexity on a text segment indicates higher likelihood of memorization.

**Implementation**:
- Calculate per-token perplexity on suspected memorized content
- Compare against perplexity on known non-training data
- Identify statistical outliers that suggest memorization
- Apply token perplexity thresholds to identify memorized code snippets [3]

**Pros**:
- Works with both open and closed models (as long as token probabilities are accessible)
- Quantitative and easy to interpret
- Doesn't require model internals or training data access
- Particularly effective for code models (e.g., Codex, StarCoder) [3]

**Cons**:
- Requires access to token probabilities (may be unavailable for some closed models)
- May produce false positives for common/predictable content
- Threshold for memorization is subjective

**Effect of Sampling Parameters on Log Probabilities**:

When using perplexity-based detection, it's important to understand how sampling parameters affect log probabilities:

- **Temperature**: Controls the randomness of token selection by scaling log probabilities.
  - Temperature < 1: Makes the probability distribution more peaked (high probabilities become higher, low become lower)
  - Temperature > 1: Flattens the distribution, making it more uniform
  - Temperature = 0: Theoretically deterministic, always selecting the highest probability token (greedy decoding)
  - Formula: `scaled_logprobs = logprobs / temperature` before applying softmax

- **Top_p (Nucleus Sampling)**: Restricts sampling to tokens comprising the top p% of probability mass.
  - Dynamically selects tokens based on cumulative probability rather than a fixed number
  - Lower values (e.g., 0.1) restrict to only the most likely tokens
  - Higher values (e.g., 0.9) allow more diversity while excluding very unlikely tokens
  - Only tokens within the selected nucleus are considered for sampling

- **Interaction Effects**:
  - At temperature = 0 with top_p = 1.0: Should consistently select highest probability token
  - At temperature = 0 with top_p < 1.0: May still be deterministic but restricted to top_p tokens
  - In practice, even with temperature = 0, implementation details can cause variability in outputs
  - Small numerical differences in token probabilities can lead to different selections when they're extremely close

For accurate perplexity-based memorization detection, it's best to use raw log probabilities directly from the model before any sampling modifications are applied.

### 2. N-gram Overlap Analysis

**Description**: Analyzes the overlap between model outputs and training corpus at the n-gram level.

**Implementation**:
- Extract n-grams from model outputs
- Compare against n-grams in reference datasets
- Calculate overlap statistics and identify exact matches

**Pros**:
- Simple to implement and understand
- Can work with black-box models
- Effective for detecting verbatim copying

**Cons**:
- Requires at least partial knowledge of training data
- May miss paraphrased or slightly modified content
- Computationally expensive for large corpora

### 3. Embedding Space Proximity

**Description**: Maps outputs to embedding space and measures distance to training examples.

**Implementation**:
- Generate embeddings for model outputs
- Compute similarity to embeddings of training data
- Flag outputs with unusually high similarity scores

**Pros**:
- Can detect semantic memorization beyond exact copying
- Works for paraphrased or restructured content
- More robust than exact string matching

**Cons**:
- Requires access to training data or representative proxies
- Higher computational requirements
- Similarity thresholds need careful calibration

### 4. Membership Inference Attacks

**Description**: Determines whether a specific data point was used in model training.

**Implementation**:
- Train a "shadow" model to classify if content was likely part of training data
- Use confidence scores and response patterns as features
- Apply statistical methods to detect outliers in model behavior

**Pros**:
- Theoretically sound with basis in privacy research
- Can work with black-box access to models
- Provides probabilistic certainty estimates

**Cons**:
- Requires substantial compute for shadow model training
- May have lower accuracy for very large models
- Needs careful experimental design

### 5. Prompt Engineering for Extraction

**Description**: Crafts specific prompts designed to trigger memorization.

**Implementation**:
- Design prefix prompts that might lead to memorized continuations
- Systematically probe model with truncated text from potential training sources
- Analyze completion patterns and consistency

**Pros**:
- Works with completely black-box models
- Can uncover unexpected memorization
- No need for training data access

**Cons**:
- Labor-intensive prompt design
- Success depends on prompt crafting skill
- May miss memorization that requires specific triggers

### 6. Gradient-Based Methods (for Open Models)

**Description**: Analyzes model parameter gradients when prompted with specific inputs.

**Implementation**:
- Measure influence of specific training examples on model parameters
- Identify parameters with unusual activation patterns
- Trace memorization to specific model components

**Pros**:
- Provides mechanistic insights into memorization
- Can locate memorization within model architecture
- Strong theoretical foundation

**Cons**:
- Only applicable to open-source models with parameter access
- Computationally intensive
- Requires deep understanding of model architecture

### 7. Benchmark Dataset Testing

**Description**: Tests models against deliberately held-out portions of known benchmarks.

**Implementation**:
- Curate examples from common benchmarks (e.g., C4, The Pile, Wikipedia)
- Prompt models with prefixes from these examples
- Measure exactness of continuation matching

**Pros**:
- Controlled experimentation environment
- Clear ground truth for evaluation
- Works with any accessible model

**Cons**:
- Limited to known benchmark datasets
- May not represent real-world memorization patterns
- Selection bias in choosing test examples

### 8. Dynamic Soft Prompt Extraction

**Description**: Trains a transformer-based generator to create input-dependent soft prompts that maximize verbatim data extraction from LLMs.

**Implementation**:
- Train a model to generate input-dependent soft prompts
- Use these prompts to maximize verbatim data extraction
- Measure using Exact Extraction Rate (ER) and Fractional ER metrics
- Adapts to input context, outperforming static prompts by up to 135.3% [2]

**Pros**:
- Adapts prompts per input to extract memorized text/code
- Highly effective for both text and code generation models
- Can work with closed models through API access

**Cons**:
- Requires training an additional transformer model
- More complex implementation than static prompts
- Computationally intensive training process

### 9. Entity-Level Memorization Analysis

**Description**: Quantifies memorization of specific entities (e.g., names, locations) using attribute-guided prompts.

**Implementation**:
- Use structured prompts like "The capital of [ENTITY] is" to test verbatim recall
- Measure success rates against dataset entropy (rare vs. common entities)
- Can achieve 61.2% accuracy in extracting unique entities from 6B-parameter models [5]

**Pros**:
- Focuses on high-value information like named entities
- Works well with both open and closed models
- Structured approach makes evaluation more standardized

**Cons**:
- May miss memorization of non-entity information
- Requires careful prompt design
- Results vary based on entity popularity/rarity

### 10. Verbatim Row/Feature Completion Tests

**Description**: Tests for tabular data memorization through row and feature completion tasks.

**Implementation**:
- Implement row completion tests: Can the model reconstruct full dataset rows?
- Feature completion tests: Does it correctly fill in missing feature values?
- Adjust thresholds based on dataset entropy (e.g., high entropy medical data vs. low entropy in Iris dataset)
- Use tools like Tabmemcheck [4]

**Pros**:
- Specialized for structured/tabular data
- Clear metrics for evaluation
- Works with both open and closed models

**Cons**:
- Limited to tabular data domains
- May not generalize to unstructured text
- Requires baseline dataset entropy calculations

## Comparison Matrix for Different Model Types

| Method | Open-Source Models | Closed API Models | Requires Training Data | Implementation Complexity |
|--------|-------------------|-------------------|------------------------|---------------------------|
| Perplexity-Based | ✅ | ⚠️ (needs probability API) | ❌ | Low |
| N-gram Overlap | ✅ | ✅ | ✅ (partial) | Medium |
| Embedding Space | ✅ | ✅ | ✅ | Medium-High |
| Membership Inference | ✅ | ✅ | ⚠️ (representative sample) | High |
| Prompt Engineering | ✅ | ✅ | ❌ | Low-Medium |
| Gradient-Based | ✅ | ❌ | ❌ | High |
| Benchmark Testing | ✅ | ✅ | ⚠️ (benchmark data) | Medium |
| Dynamic Soft Prompts | ✅ | ✅ | ❌ | High |
| Entity-Level Analysis | ✅ | ✅ | ⚠️ (entity list) | Medium |
| Row/Feature Completion | ✅ | ✅ | ✅ (tabular data) | Medium |

## Implementation Considerations

For a comprehensive memorization detection tool, we recommend:

1. **Multi-method approach**: Combine multiple detection methods for higher confidence
2. **Adaptable thresholds**: Allow customizable sensitivity settings
3. **Model-specific calibration**: Account for different model architectures and sizes
4. **Statistical validation**: Include significance testing to reduce false positives
5. **Interpretable reports**: Generate clear visualizations and explanations of results
6. **Data entropy assessment**: Calculate and account for dataset uniqueness when setting thresholds [4][5]
7. **Context-aware evaluation**: Adjust tests based on model size and training conditions [1][6]

## Mitigation Strategies

Research suggests several approaches to reduce memorization risks:

1. **Data Deduplication**: Removing duplicates from training sets reduces memorization by ~30% [3][6]
2. **Forgetting Baselines**: Monitor tokens forgotten during training—larger models forget less [1]
3. **Entropy Weighting**: Adjust thresholds based on dataset uniqueness [4][5]
4. **Differential Privacy**: Apply DP techniques during model training to limit memorization [6]

## Ethical Considerations

When using this tool, consider:

- Responsible disclosure if potentially sensitive memorized content is discovered
- Balancing evaluation needs with potential privacy implications
- Recognizing that some memorization is expected and even necessary
- Proper attribution when testing with copyright materials

> **Note**: Detailed implementation ideas and code examples are available in the [TOOLDESIGN.md](TOOLDESIGN.md) file.

## LLM Benchmarks for Memorization Testing

When implementing perplexity-based methods and other memorization detection techniques, the following benchmarks can be used, ordered approximately by dataset size:

### Small Datasets
- **HellaSwag** (~70K examples) - Common sense reasoning and language understanding
- **MMLU** (Multiple-choice, ~15K questions) - Multi-domain knowledge across 57 subjects
- **TruthfulQA** (~800 questions) - Evaluates model truthfulness and misinformation
- **ARC** (Challenge: ~2K questions, Easy: ~5K questions) - Grade-school science questions
- **MedQA-USMLE** (~12K questions) - Medical licensing exam questions with 4 options per question

### Medium Datasets
- **SQuAD** (~100K questions) - Stanford Question Answering Dataset for reading comprehension
- **Natural Questions** (~300K examples) - Real Google search queries with answers
- **GLUE** (~75K-120K examples across tasks) - General Language Understanding Evaluation
- **SuperGLUE** (~150K examples across tasks) - More challenging version of GLUE

### Large Datasets
- **C4** (Common Crawl, ~365M documents) - Colossal Clean Crawled Corpus
- **The Pile** (~825GB of diverse text) - Large-scale text corpus for language modeling
- **LAMBADA** (~10K passages, drawn from large corpus) - Language modeling benchmark
- **WikiText** (103M tokens) - Long-form text extracted from Wikipedia

### Very Large Datasets
- **RedPajama** (~1.2T tokens) - Open reproduction of LLaMA training data
- **The Stack** (~6TB code corpus) - Programming code dataset across multiple languages
- **ROOTS** (~1.6T tokens) - Multilingual dataset with diverse content

## LLM APIs for Probability Extraction

The following APIs can be used to extract token probabilities needed for perplexity-based memorization detection:

1. **OpenAI API**
   - Models: GPT-4, GPT-3.5-Turbo
   - Features: Supports `logprobs` parameter in completions API
   - Documentation: Can return top logprobs for most likely tokens and chosen tokens

2. **Anthropic Claude API**
   - Models: Claude series (Claude 3 Opus, Sonnet, Haiku)
   - Features: Supports token probabilities through dedicated endpoints
   - Documentation: Provides probabilities for generated tokens

3. **Hugging Face Inference API**
   - Models: Access to thousands of open models
   - Features: Full logprob access for most models
   - Documentation: Returns detailed token probability distributions

4. **Google Gemini API**
   - Models: Gemini series
   - Features: Supports candidate token scoring and logprobs
   - Documentation: Can return top token probabilities

5. **Self-hosted options**
   - Implementation: Using libraries like Transformers, vLLM
   - Models: Llama 2, Mistral, Falcon, etc.
   - Features: Full access to all token probabilities and model internals

When implementing perplexity-based methods, consider:
- Token probability format (log vs. raw probabilities)
- API rate limits and costs
- Tokenization differences between models
- Availability of full probability distributions vs. top-k probabilities

## Future Directions

Future enhancements could include:

- Integration with model training pipelines for continuous monitoring
- Expanded benchmark datasets specifically designed for memorization testing
- Development of differential privacy metrics for quantifying memorization risk
- Extension to multimodal models beyond text
- Implementation of dynamic soft prompt extraction for enhanced detection [2]
- Exploration of model editing techniques to selectively remove memorized content [6]

## References

[1] Ramasesh, V., et al. (2022). Effect of scale on catastrophic forgetting in neural networks. NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/file/fa0509f4dab6807e2cb465715bf2d249-Paper-Conference.pdf

[2] Li, Y., et al. (2024). Extracting Training Data from Large Language Models with Dynamic Soft Prompts. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.546.pdf

[3] Carlini, N., et al. (2023). Extracting Training Data from Large Language Models. arXiv preprint. https://arxiv.org/html/2308.09932v2

[4] InterpretML. (2024). LLM-Tabular-Memorization-Checker. GitHub Repository. https://github.com/interpretml/LLM-Tabular-Memorization-Checker

[5] Wang, X., et al. (2024). Entity-Level Memorization in Language Models. AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/29948/31657

[6] Zhang, H., et al. (2024). Quantifying Memorization Across Neural Language Models. OpenReview. https://openreview.net/forum?id=TatRHT_1cK
