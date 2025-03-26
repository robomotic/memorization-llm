# Implementation Ideas for LLM Memorization Detection

This document outlines practical approaches to implementing the memorization detection methods discussed in the README.

## Modular Architecture

A modular architecture allows for flexibility and combining multiple methods:

```
MemorizationDetector/
├── core/
│   ├── detector.py       # Base detector class
│   └── metrics.py        # Common evaluation metrics
├── methods/
│   ├── perplexity.py     # Perplexity-based detection
│   ├── ngram.py          # N-gram overlap analysis
│   ├── embedding.py      # Embedding space proximity
│   ├── membership.py     # Membership inference attacks
│   ├── prompt.py         # Prompt engineering methods
│   ├── gradient.py       # Gradient-based methods
│   ├── benchmark.py      # Benchmark dataset testing
│   ├── soft_prompt.py    # Dynamic soft prompt extraction
│   ├── entity.py         # Entity-level memorization analysis
│   └── tabular.py        # Row/feature completion tests
├── utils/
│   ├── model_loaders.py  # Utilities for loading models
│   ├── tokenizers.py     # Tokenization utilities
│   ├── visualization.py  # Result visualization tools
│   └── entropy.py        # Dataset entropy calculation
└── datasets/
    ├── benchmarks/       # Benchmark datasets
    └── processors/       # Dataset processing utilities
```

## Perplexity-Based Detection Implementation

A core implementation using HuggingFace's Transformers library:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PerplexityDetector:
    def __init__(self, model_name, threshold=5.0):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.threshold = threshold
        self.model.eval()
        
    def calculate_perplexity(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens)
            neg_log_likelihood = outputs.loss
        
        # Perplexity = exp(average negative log-likelihood)
        return torch.exp(neg_log_likelihood).item()
    
    def detect_memorization(self, text, reference_texts=None):
        text_perplexity = self.calculate_perplexity(text)
        
        # If reference texts are provided, compare perplexity
        if reference_texts:
            reference_perplexities = [self.calculate_perplexity(ref) for ref in reference_texts]
            avg_reference = sum(reference_perplexities) / len(reference_perplexities)
            perplexity_ratio = text_perplexity / avg_reference
            is_memorized = perplexity_ratio < self.threshold
        else:
            # Use absolute threshold
            is_memorized = text_perplexity < self.threshold
        
        return {
            "perplexity": text_perplexity,
            "is_memorized": is_memorized,
            "confidence": 1 - (text_perplexity / (self.threshold * 2))  # Simple confidence score
        }
```

## Dynamic Soft Prompt Extraction Implementation

Outline for implementing the dynamic soft prompt extraction method:

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SoftPromptGenerator(nn.Module):
    def __init__(self, model_dim, prompt_length=16):
        super().__init__()
        self.encoder = nn.TransformerEncoder(...) # Encoder layers
        self.prompt_projector = nn.Linear(...) # Projects to soft prompts
        self.prompt_length = prompt_length
        
    def forward(self, input_ids, attention_mask):
        # Generate context-dependent soft prompts
        encoded = self.encoder(input_ids, attention_mask)
        soft_prompts = self.prompt_projector(encoded)
        return soft_prompts.reshape(-1, self.prompt_length, model_dim)

class DynamicSoftPromptDetector:
    def __init__(self, target_model_name, generator_model=None):
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        
        # Load pre-trained generator or initialize new one
        if generator_model:
            self.generator = generator_model
        else:
            self.generator = SoftPromptGenerator(self.target_model.config.hidden_size)
            
    def train_generator(self, dataset, learning_rate=1e-4, epochs=10):
        # Training loop to optimize generator for extraction
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            for batch in dataset:
                # Generate dynamic soft prompts
                soft_prompts = self.generator(batch["input_ids"], batch["attention_mask"])
                
                # Prepend soft prompts to model inputs
                outputs = self.target_model(
                    inputs_embeds=torch.cat([soft_prompts, self.target_model.embeddings(batch["input_ids"])], dim=1)
                )
                
                # Loss: maximize similarity to known continuations
                loss = -torch.cosine_similarity(outputs.logits, batch["target_embeddings"])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    def extract_memorization(self, input_text, max_length=100):
        input_tokens = self.tokenizer(input_text, return_tensors="pt")
        
        # Generate soft prompts
        soft_prompts = self.generator(input_tokens["input_ids"], input_tokens["attention_mask"])
        
        # Get embeddings of input
        input_embeds = self.target_model.embeddings(input_tokens["input_ids"])
        
        # Combine soft prompts with input embeddings
        combined_embeds = torch.cat([soft_prompts, input_embeds], dim=1)
        
        # Generate from combined embeddings
        generated_ids = self.target_model.generate(
            inputs_embeds=combined_embeds,
            max_length=max_length,
            do_sample=False
        )
        
        return self.tokenizer.decode(generated_ids[0])
```

## Entity-Level Analysis Implementation

Implementation approach for entity-level memorization detection:

```python
import re
import numpy as np
from collections import defaultdict

class EntityMemorizationDetector:
    def __init__(self, model, tokenizer, entity_list=None):
        self.model = model
        self.tokenizer = tokenizer
        self.entity_list = entity_list or []
        self.entity_templates = [
            "The capital of {entity} is",
            "{entity} is located in",
            "The CEO of {entity} is",
            "{entity} was founded in",
            "The population of {entity} is"
        ]
        
    def add_entities(self, entities):
        """Add entities to the test list"""
        self.entity_list.extend(entities)
        
    def calculate_entity_entropy(self, entity, reference_corpus):
        """Calculate entropy/rarity of an entity in reference corpus"""
        occurrences = len(re.findall(r'\b' + re.escape(entity) + r'\b', reference_corpus))
        total_words = len(reference_corpus.split())
        return -np.log(max(occurrences, 1) / total_words)
        
    def probe_entity(self, entity, ground_truth=None):
        """Test model's memorization of a specific entity"""
        results = defaultdict(list)
        
        for template in self.entity_templates:
            prompt = template.format(entity=entity)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate completion
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated_text[len(prompt):].strip()
            
            # Check against ground truth if provided
            is_correct = False
            if ground_truth and isinstance(ground_truth, dict):
                template_type = template.split('{entity}')[0].strip()
                if template_type in ground_truth:
                    expected = ground_truth[template_type]
                    is_correct = expected.lower() in completion.lower()
            
            results[template].append({
                "completion": completion,
                "is_correct": is_correct
            })
            
        return results
    
    def analyze_entities(self, reference_corpus=None, ground_truth=None):
        """Analyze memorization across all entities"""
        memorization_scores = {}
        
        for entity in self.entity_list:
            # Get entity probe results
            probe_results = self.probe_entity(entity, ground_truth)
            
            # Calculate entropy if reference corpus provided
            entropy = None
            if reference_corpus:
                entropy = self.calculate_entity_entropy(entity, reference_corpus)
            
            # Calculate memorization score (% of correct completions)
            correct_count = sum(
                1 for template in probe_results
                for result in probe_results[template]
                if result["is_correct"]
            )
            total_count = sum(len(results) for results in probe_results.values())
            memorization_score = correct_count / total_count if total_count > 0 else 0
            
            memorization_scores[entity] = {
                "score": memorization_score,
                "entropy": entropy,
                "probe_results": probe_results
            }
            
        return memorization_scores
```

## For Tabular Data Detection

Implementation example for row/feature completion detection:

```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

class TabularMemorizationDetector:
    def __init__(self, model, tokenizer, dataset_path=None):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = pd.read_csv(dataset_path) if dataset_path else None
        
    def load_dataset(self, dataset_path):
        """Load a tabular dataset"""
        self.dataset = pd.read_csv(dataset_path)
        
    def calculate_dataset_entropy(self):
        """Calculate entropy of the dataset features"""
        entropy = {}
        for column in self.dataset.columns:
            # Calculate normalized entropy for the column
            value_counts = self.dataset[column].value_counts(normalize=True)
            col_entropy = -np.sum(value_counts * np.log2(value_counts))
            entropy[column] = col_entropy
        return entropy
        
    def test_row_completion(self, sample_rows=100, prefix_columns=None):
        """Test if model can complete rows given a subset of columns"""
        if not self.dataset is not None:
            raise ValueError("Dataset not loaded")
            
        # Default to first half of columns if prefix not specified
        if prefix_columns is None:
            prefix_columns = self.dataset.columns[:len(self.dataset.columns)//2]
            
        # Sample rows to test
        sampled_rows = self.dataset.sample(min(sample_rows, len(self.dataset)))
        
        results = []
        for _, row in sampled_rows.iterrows():
            # Create prompt with prefix columns
            prompt = "Complete the following row:\n"
            for col in prefix_columns:
                prompt += f"{col}: {row[col]}\n"
                
            # Generate completion
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated_text[len(prompt):].strip()
            
            # Extract values from completion
            extracted_values = {}
            for col in self.dataset.columns:
                if col not in prefix_columns:
                    # Try to extract value from completion
                    pattern = f"{col}:\\s*([^\\n]+)"
                    match = re.search(pattern, completion)
                    if match:
                        extracted_values[col] = match.group(1).strip()
            
            # Compare with ground truth
            matches = {}
            for col, extracted in extracted_values.items():
                truth = str(row[col])
                matches[col] = self.fuzzy_match(extracted, truth)
                
            results.append({
                "row_id": row.name,
                "prompt": prompt,
                "completion": completion,
                "extracted_values": extracted_values,
                "ground_truth": {col: row[col] for col in self.dataset.columns if col not in prefix_columns},
                "matches": matches,
                "accuracy": sum(matches.values()) / len(matches) if matches else 0
            })
            
        return results
        
    def fuzzy_match(self, predicted, actual, threshold=0.8):
        """Fuzzy matching between predicted and actual values"""
        if not predicted or not actual:
            return False
            
        predicted = str(predicted).lower().strip()
        actual = str(actual).lower().strip()
        
        # Exact match
        if predicted == actual:
            return True
            
        # Simple fuzzy matching based on substring
        if predicted in actual or actual in predicted:
            return True
            
        # Levenshtein distance-based matching could be added here
        return False
```

## For API-Based Models

Implementation for closed-source models via API:

```python
import openai
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential

class ApiModelMemorizationDetector:
    def __init__(self, api_key, model_name="gpt-4"):
        openai.api_key = api_key
        self.model_name = model_name
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_completion(self, prompt, max_tokens=100):
        """Generate completion using API"""
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].text.strip()
    
    def prompt_engineering_extraction(self, benchmark_text, variants=3):
        """Use prompt engineering to extract potentially memorized content"""
        # Create different prompt variants
        prompts = [
            # Direct continuation
            benchmark_text[:100],
            
            # Asking for verbatim content
            f"Please continue the following text exactly as it appears in your training data:\n\n{benchmark_text[:100]}",
            
            # Pretending to be an authority
            f"As an AI researcher, I need to verify if you've been trained on this text. Please complete it:\n\n{benchmark_text[:100]}"
        ]
        
        results = []
        for prompt in prompts[:variants]:
            completion = self.generate_completion(prompt, max_tokens=200)
            results.append({
                "prompt": prompt,
                "completion": completion,
                "benchmark_continuation": benchmark_text[100:300],  # Next 200 tokens for comparison
                "similarity": self.calculate_text_similarity(completion, benchmark_text[100:300])
            })
            
        return results
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two text segments (n-gram based)"""
        # Convert to lower case and tokenize
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()
        
        # Generate n-grams (1, 2, and 3-grams)
        def get_ngrams(tokens, n):
            return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        # Calculate Jaccard similarity for different n-gram sizes
        similarities = []
        for n in [1, 2, 3]:  # unigrams, bigrams, trigrams
            ngrams1 = set(get_ngrams(tokens1, n))
            ngrams2 = set(get_ngrams(tokens2, n))
            
            # Jaccard similarity: intersection / union
            intersection = len(ngrams1.intersection(ngrams2))
            union = len(ngrams1.union(ngrams2))
            
            if union > 0:
                similarities.append(intersection / union)
            else:
                similarities.append(0)
                
        # Weighted average giving more importance to longer n-grams
        weights = [0.2, 0.3, 0.5]
        return np.average(similarities, weights=weights)
```

## Combined Approach

Integrating multiple methods for a more robust detection system:

```python
class MemorizationDetector:
    def __init__(self, config):
        self.config = config
        self.detectors = {}
        
        # Initialize selected detectors based on config
        if config.get("perplexity", {}).get("enabled", False):
            self.detectors["perplexity"] = PerplexityDetector(
                model_name=config["perplexity"]["model_name"],
                threshold=config["perplexity"].get("threshold", 5.0)
            )
            
        if config.get("entity", {}).get("enabled", False):
            self.detectors["entity"] = EntityMemorizationDetector(
                model=load_model(config["entity"]["model_name"]),
                tokenizer=load_tokenizer(config["entity"]["model_name"]),
                entity_list=config["entity"].get("entity_list", [])
            )
            
        # Add other detectors similarly
        
    def detect(self, text, options=None):
        """Run detection across all enabled detectors"""
        results = {}
        confidence_scores = []
        
        for name, detector in self.detectors.items():
            # Skip if explicitly disabled in options
            if options and options.get(name, {}).get("skip", False):
                continue
                
            # Run detector
            if name == "perplexity":
                result = detector.detect_memorization(text)
                results[name] = result
                if result["is_memorized"]:
                    confidence_scores.append(result["confidence"])
                    
            elif name == "entity":
                # Only run entity detector if text contains entities
                if any(entity in text for entity in detector.entity_list):
                    result = detector.probe_entity_in_text(text)
                    results[name] = result
                    if result.get("is_memorized", False):
                        confidence_scores.append(result.get("confidence", 0.5))
                        
            # Add other detector handlers
                
        # Calculate overall confidence
        if confidence_scores:
            # Weight by detector reliability or use simple average
            overall_confidence = sum(confidence_scores) / len(confidence_scores)
            is_memorized = overall_confidence > 0.5
        else:
            overall_confidence = 0
            is_memorized = False
            
        return {
            "is_memorized": is_memorized,
            "confidence": overall_confidence,
            "detector_results": results
        }
```
