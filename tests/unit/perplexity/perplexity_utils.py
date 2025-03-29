"""
Utilities for token-level perplexity calculations and analysis.
"""
import math
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import tiktoken
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerplexityAnalyzer:
    """Analyzes perplexity at the token level, supporting various tokenizers and models."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """Initialize the perplexity analyzer.
        
        Args:
            model_name: Name of the model to use (determines tokenizer)
            api_key: API key for model access (if needed)
        """
        self.model_name = model_name
        self.api_key = api_key
        
        # Map models to their corresponding tokenizers
        model_to_encoding = {
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4": "cl100k_base",
            "text-davinci-003": "p50k_base",
            "code-davinci-002": "p50k_base",
            "llama": "llama",
            "custom": None  # For custom tokenizers
        }
        
        # Get the encoding for this model
        if model_name in model_to_encoding and model_to_encoding[model_name]:
            try:
                self.tokenizer = tiktoken.get_encoding(model_to_encoding[model_name])
                logger.info(f"Using {model_to_encoding[model_name]} tokenizer for {model_name}")
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                self.tokenizer = None
        else:
            logger.warning(f"No predefined tokenizer for {model_name}, using default cl100k_base")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize input text into token IDs.
        
        Args:
            text: The input text to tokenize
            
        Returns:
            List of token IDs
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not available")
        
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs into human-readable tokens.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of decoded token strings
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not available")
        
        return [self.tokenizer.decode([token_id]) for token_id in token_ids]
    
    def calculate_token_logprobs(self, text: str, api_url: Optional[str] = None) -> Dict[str, Any]:
        """Calculate log probabilities for each token in the text.
        
        Args:
            text: Input text to analyze
            api_url: Optional API URL for token probability calculation
            
        Returns:
            Dictionary with token logprobs and perplexity information
        """
        # This is a placeholder. In a real implementation, you would:
        # 1. Call an LLM API that supports logprobs
        # 2. Get token-by-token logprobs
        # 3. Properly format and return the results
        
        # For now, we'll use a simple mock implementation
        token_ids = self.tokenize(text)
        tokens = self.decode_tokens(token_ids)
        
        # Generate mock logprobs (would be replaced with real API calls)
        # More probable (common) tokens would have higher (less negative) logprobs
        logprobs = []
        
        for token in tokens:
            # Simulate token probabilities - more common tokens have higher logprobs
            if token.strip() in [" ", "the", "a", "is", "of", "and", "in", "to", "that"]:
                # Common tokens have higher (less negative) logprobs
                logprob = np.random.uniform(-2.0, -1.0)
            elif token.strip() in ["Paris", "France", "capital", "London", "Berlin"]:
                # Factual tokens could have medium probabilities
                logprob = np.random.uniform(-3.5, -2.0)
            else:
                # Other tokens have lower (more negative) logprobs
                logprob = np.random.uniform(-6.0, -3.5)
            
            logprobs.append(logprob)
        
        # Calculate perplexity from logprobs
        avg_logprob = sum(logprobs) / len(logprobs)
        perplexity = math.exp(-avg_logprob)
        
        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "logprobs": logprobs,
            "avg_logprob": avg_logprob,
            "perplexity": perplexity
        }
    
    def calculate_token_level_perplexity(self, text: str) -> List[Dict[str, Any]]:
        """Calculate perplexity iteratively at each token position.
        
        This calculates perplexity for each subwindow from the beginning to position i,
        showing how perplexity changes as the text progresses.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of perplexity results for each token position
        """
        results = []
        token_ids = self.tokenize(text)
        token_count = len(token_ids)
        
        # For each position, calculate perplexity of the prefix (from start to position i)
        for i in range(1, token_count + 1):
            # Get the prefix from position 0 to position i-1 (inclusive)
            prefix_ids = token_ids[:i]
            prefix_text = self.tokenizer.decode(prefix_ids)
            
            # Calculate perplexity for this prefix
            prefix_results = self.calculate_token_logprobs(prefix_text)
            
            results.append({
                "position": i-1,  # Adjust position to match 0-indexed tokens
                "prefix_text": prefix_text,
                "token_count": len(prefix_ids),
                "avg_logprob": prefix_results["avg_logprob"],
                "perplexity": prefix_results["perplexity"],
                "tokens": prefix_results["tokens"],
                "logprobs": prefix_results["logprobs"]
            })
        
        return results
    
    def compare_perplexities(self, text_list: List[str]) -> Dict[str, Any]:
        """Compare perplexities across multiple texts.
        
        Args:
            text_list: List of texts to compare
            
        Returns:
            Dictionary with comparative perplexity information
        """
        results = []
        
        for text in text_list:
            text_perplexity = self.calculate_token_logprobs(text)
            results.append({
                "text": text,
                "perplexity": text_perplexity["perplexity"],
                "avg_logprob": text_perplexity["avg_logprob"],
                "token_count": len(text_perplexity["tokens"])
            })
        
        # Sort by perplexity (lowest to highest)
        sorted_results = sorted(results, key=lambda x: x["perplexity"])
        
        return {
            "texts": len(text_list),
            "results": sorted_results,
            "lowest_perplexity": sorted_results[0] if sorted_results else None,
            "highest_perplexity": sorted_results[-1] if sorted_results else None,
            "avg_perplexity": sum(r["perplexity"] for r in results) / len(results) if results else 0
        }
