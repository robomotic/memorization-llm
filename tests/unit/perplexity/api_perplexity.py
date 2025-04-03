"""
Advanced perplexity testing with real API integration.
This module connects to actual LLM APIs to get token probabilities.
"""
import os
import json
import time
import logging
import requests
import numpy as np
import tiktoken
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APIPerplexityAnalyzer:
    """Calculates perplexity using token probabilities from real LLM APIs."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, api_backend: str = "openai"):
        """Initialize the API perplexity analyzer.
        
        Args:
            model_name: Model to use for perplexity calculation
            api_key: API key for the model provider
            api_backend: API backend to use (openai, azure, or openrouter)
        """
        self.model_name = model_name
        self.api_backend = api_backend.lower()
        
        # Set API key based on the backend if not provided
        if not api_key:
            if self.api_backend == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.api_backend == "azure":
                self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            elif self.api_backend == "openrouter":
                self.api_key = os.getenv("OPENROUTER_API_KEY")
            else:
                raise ValueError(f"Invalid API backend: {api_backend}. Must be 'openai', 'azure', or 'openrouter'.")
        else:
            self.api_key = api_key
        
        # Get additional configuration for Azure
        if self.api_backend == "azure":
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            
            if not all([self.api_key, self.azure_endpoint, self.azure_deployment]):
                missing = []
                if not self.api_key: missing.append("AZURE_OPENAI_API_KEY")
                if not self.azure_endpoint: missing.append("AZURE_OPENAI_ENDPOINT")
                if not self.azure_deployment: missing.append("AZURE_OPENAI_DEPLOYMENT")
                logger.warning(f"Missing Azure OpenAI configuration: {', '.join(missing)}")
        
        # Get OpenRouter base URL if using OpenRouter
        if self.api_backend == "openrouter":
            self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        if not self.api_key:
            logger.warning(f"No API key provided for {self.api_backend}. Set appropriate API key in .env file or pass it explicitly.")
        
        # Map models to their corresponding tokenizers
        model_to_encoding = {
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4": "cl100k_base",
            "text-davinci-003": "p50k_base",
            "code-davinci-002": "p50k_base",
        }
        
        # Get the encoding for this model
        try:
            encoding_name = model_to_encoding.get(model_name, "cl100k_base")
            self.tokenizer = tiktoken.get_encoding(encoding_name)
            logger.info(f"Using {encoding_name} tokenizer for {model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the model's tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs into readable strings.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of decoded tokens
        """
        return [self.tokenizer.decode([token_id]) for token_id in token_ids]
    
    def get_token_probabilities(self, text: str) -> Dict[str, Any]:
        """Get token probabilities from the LLM API.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with token probabilities and perplexity information
        """
        if not self.api_key:
            raise ValueError("API key not provided")
        
        # Configure API request based on the backend
        if self.api_backend == "openai":
            return self._get_openai_token_probabilities(text)
        elif self.api_backend == "azure":
            return self._get_azure_token_probabilities(text)
        elif self.api_backend == "openrouter":
            return self._get_openrouter_token_probabilities(text)
        else:
            raise ValueError(f"Unsupported API backend: {self.api_backend}")
    
    def _get_openai_token_probabilities(self, text: str) -> Dict[str, Any]:
        """Get token probabilities from the OpenAI API."""
        # Check if the model is a chat model (gpt-3.5-turbo, gpt-4, etc.)
        is_chat_model = any(chat_model in self.model_name.lower() for chat_model in ["gpt-3.5", "gpt-4", "turbo"])
        
        if is_chat_model:
            return self._get_openai_chat_token_probabilities(text)
        else:
            return self._get_openai_completion_token_probabilities(text)
    
    def _get_openai_completion_token_probabilities(self, text: str) -> Dict[str, Any]:
        """Get token probabilities from the OpenAI API using the completions endpoint."""
        api_url = "https://api.openai.com/v1/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # For OpenAI API completions endpoint
        payload = {
            "model": self.model_name,
            "prompt": text,
            "max_tokens": 1,  # Just need logprobs, not actually generating
            "temperature": 0.0,  # Deterministic for consistency
            "logprobs": 5,  # Get logprobs for top tokens
            "echo": True  # Return the prompt with logprobs
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Process the response to extract token logprobs
            if 'choices' not in data or not data['choices']:
                raise ValueError(f"Invalid API response format: {data}")
            
            choice = data['choices'][0]
            if 'logprobs' not in choice:
                raise ValueError(f"No logprobs in API response: {choice}")
            
            logprobs_data = choice['logprobs']
            tokens = logprobs_data.get('tokens', [])
            token_logprobs = logprobs_data.get('token_logprobs', [])
            top_logprobs = logprobs_data.get('top_logprobs', [])
            
            return self._process_token_probabilities(tokens, token_logprobs, top_logprobs)
        except Exception as e:
            logger.error(f"Error getting token probabilities from OpenAI completions API: {e}")
            raise
    
    def _get_openai_chat_token_probabilities(self, text: str) -> Dict[str, Any]:
        """Get token probabilities from the OpenAI API using the chat completions endpoint."""
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # For OpenAI API chat completions endpoint
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Process the chat completions response
            return self._process_openai_chat_response(data)
        except Exception as e:
            logger.error(f"Error getting token probabilities from OpenAI chat API: {e}")
            raise
    
    def _process_openai_chat_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the OpenAI chat completions API response."""
        # Process the response to extract token logprobs
        if 'choices' not in data or not data['choices']:
            raise ValueError(f"Invalid API response format: {data}")
        
        choice = data['choices'][0]
        if 'logprobs' not in choice:
            raise ValueError(f"No logprobs in API response: {choice}")
        
        # Extract logprobs data (similar to OpenRouter format)
        logprobs_data = choice['logprobs']
        content_array = logprobs_data.get('content', [])
        
        if not content_array:
            raise ValueError(f"No content in logprobs data: {logprobs_data}")
        
        # Extract tokens and logprobs from the content array
        tokens = [item.get('token', '') for item in content_array]
        token_logprobs = [item.get('logprob', None) for item in content_array]
        
        # Extract top_logprobs
        top_logprobs = [item.get('top_logprobs', []) for item in content_array]
        
        return self._process_token_probabilities(tokens, token_logprobs, top_logprobs)
    
    def _get_azure_token_probabilities(self, text: str) -> Dict[str, Any]:
        """Get token probabilities from the Azure OpenAI API."""
        if not all([self.azure_endpoint, self.azure_deployment]):
            raise ValueError("Azure OpenAI endpoint and deployment must be provided")
        
        api_url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/completions?api-version={self.azure_api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        # For Azure OpenAI API completions endpoint
        payload = {
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 5,
            "echo": True
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Process the response to extract token logprobs
            if 'choices' not in data or not data['choices']:
                raise ValueError(f"Invalid API response format: {data}")
            
            choice = data['choices'][0]
            if 'logprobs' not in choice:
                raise ValueError(f"No logprobs in API response: {choice}")
            
            logprobs_data = choice['logprobs']
            tokens = logprobs_data.get('tokens', [])
            token_logprobs = logprobs_data.get('token_logprobs', [])
            top_logprobs = logprobs_data.get('top_logprobs', [])
            
            return self._process_token_probabilities(tokens, token_logprobs, top_logprobs)
        except Exception as e:
            logger.error(f"Error getting token probabilities from Azure OpenAI API: {e}")
            raise
    
    def _get_openrouter_token_probabilities(self, text: str) -> Dict[str, Any]:
        """Get token probabilities from the OpenRouter API."""
        api_url = f"{self.openrouter_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/robomotic/MemorizationLLM",
            "X-Title": "Memorization Detection"
        }
        
        # For OpenRouter API chat completions endpoint
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5
        }
        
        # Tokenize the input text using the appropriate model tokenizer
        text_token_ids = self.tokenize(text)
        text_tokens = self.decode_tokens(text_token_ids)
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Process the response to extract token logprobs
            if 'choices' not in data or not data['choices']:
                raise ValueError(f"Invalid API response format: {data}")
            
            choice = data['choices'][0]
            if 'logprobs' not in choice:
                # If the model doesn't support logprobs, use a fallback method
                logger.warning(f"Model {self.model_name} doesn't support logprobs via OpenRouter. Using fallback method.")
                return self._fallback_token_probabilities(text)
            
            logprobs_data = choice['logprobs']
            
            # OpenRouter returns logprobs.content as an array of objects
            # Each object has token, logprob, and top_logprobs properties
            content_array = logprobs_data.get('content', [])
            
            if not content_array:
                logger.warning(f"No content in logprobs data from OpenRouter API: {logprobs_data}")
                return self._fallback_token_probabilities(text)
            
            # Extract tokens and logprobs from the content array
            tokens = [item.get('token', '') for item in content_array]
            token_logprobs = [item.get('logprob', None) for item in content_array]
            
            # Extract top_logprobs (this is a list of lists of objects)
            top_logprobs = [item.get('top_logprobs', []) for item in content_array]
            
            return self._process_token_probabilities(tokens, token_logprobs, top_logprobs, text_tokens=text_tokens)
        except Exception as e:
            logger.error(f"Error getting token probabilities from OpenRouter API: {e}")
            raise
    
    def _fallback_token_probabilities(self, text: str) -> Dict[str, Any]:
        """Fallback method when logprobs are not available."""
        # This is a simplified fallback that estimates perplexity without token probabilities
        # In a real implementation, you might want to use a different model or approach
        token_ids = self.tokenize(text)
        tokens = self.decode_tokens(token_ids)
        
        # Generate synthetic logprobs (this is just an approximation)
        token_count = len(tokens)
        token_logprobs = [-2.0] * token_count  # Default logprob of -2.0 (relatively low probability)
        
        # Make common tokens more probable
        common_tokens = [" ", "the", "a", "an", "is", "are", "to", "of", "and", "in"]
        for i, token in enumerate(tokens):
            if token.lower() in common_tokens:
                token_logprobs[i] = -1.0  # Higher probability for common tokens
        
        # Calculate perplexity from these synthetic logprobs
        return self._process_token_probabilities(tokens, token_logprobs, None)
    
    def _process_token_probabilities(self, tokens, token_logprobs, top_logprobs, text_tokens=None, prev_tokens:int=0):
        # Filter out None values (first token might have None logprob)
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
        
        # Use the length of text_tokens if provided, otherwise use prev_tokens
        token_count = len(text_tokens) if text_tokens is not None else prev_tokens + len(valid_logprobs)
        
        # Calculate perplexity from valid logprobs
        if valid_logprobs and token_count > 0:
            avg_logprob = sum(valid_logprobs) / token_count
            perplexity = np.exp(-avg_logprob)
        else:
            avg_logprob = 0.0
            perplexity = float('inf')
        
        result = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "avg_logprob": avg_logprob,
            "perplexity": perplexity
        }
        
        # Add text_tokens to the result if available
        if text_tokens is not None:
            result["text_tokens"] = text_tokens
        
        return result
    
    def progressive_perplexity(self, text: str) -> List[Dict[str, Any]]:
        """Calculate perplexity progressively for subwindows of text.
        
        For each token position, calculate the perplexity of the suffix from that position.
        Shows how perplexity changes as more prefix context is added.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of perplexity results for each token position
        """
        token_ids = self.tokenize(text)
        results = []
        
        # For each position, calculate perplexity of the prefix (from start to position i)
        for i in range(1, len(token_ids) + 1):
            # Get the prefix from position 0 to position i-1 (inclusive)
            prefix_ids = token_ids[:i]
            prefix_text = self.tokenizer.decode(prefix_ids)
            
            # Get perplexity for this prefix
            try:
                prefix_result = self.get_token_probabilities(prefix_text)
                
                results.append({
                    "position": i-1,  # Adjust position to match 0-indexed tokens
                    "prefix_text": prefix_text,
                    "token_count": len(prefix_ids),
                    "perplexity": prefix_result["perplexity"],
                    "avg_logprob": prefix_result["avg_logprob"],
                    "token_logprobs": prefix_result["token_logprobs"],
                    "top_logprobs": prefix_result["top_logprobs"]
                })
            except Exception as e:
                logger.error(f"Error calculating perplexity for position {i}: {e}")
                # Continue with next position
            
            # Sleep to avoid rate limiting
            time.sleep(0.5)
        
        return results
    
    def analyze_perplexity_contrast(self, known_texts: List[str], random_texts: List[str]) -> Dict[str, Any]:
        """Compare perplexity between known facts and random strings.
        
        Args:
            known_texts: List of known facts or common phrases
            random_texts: List of random high-entropy strings
            
        Returns:
            Dictionary with comparative analysis
        """
        known_results = []
        random_results = []
        
        # Process known texts
        for text in known_texts:
            try:
                result = self.get_token_probabilities(text)
                known_results.append({
                    "text": text,
                    "perplexity": result["perplexity"],
                    "avg_logprob": result["avg_logprob"]
                })
                # Sleep to avoid rate limiting
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error processing known text: {e}")
        
        # Process random texts
        for text in random_texts:
            try:
                result = self.get_token_probabilities(text)
                random_results.append({
                    "text": text[:30] + "..." if len(text) > 30 else text,  # Truncate for display
                    "perplexity": result["perplexity"],
                    "avg_logprob": result["avg_logprob"]
                })
                # Sleep to avoid rate limiting
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error processing random text: {e}")
        
        # Calculate aggregate statistics
        known_perplexities = [r["perplexity"] for r in known_results]
        random_perplexities = [r["perplexity"] for r in random_results]
        
        avg_known = sum(known_perplexities) / len(known_perplexities) if known_perplexities else 0
        avg_random = sum(random_perplexities) / len(random_perplexities) if random_perplexities else 0
        
        return {
            "known_texts": {
                "results": known_results,
                "avg_perplexity": avg_known,
                "min_perplexity": min(known_perplexities) if known_perplexities else 0,
                "max_perplexity": max(known_perplexities) if known_perplexities else 0
            },
            "random_texts": {
                "results": random_results,
                "avg_perplexity": avg_random,
                "min_perplexity": min(random_perplexities) if random_perplexities else 0,
                "max_perplexity": max(random_perplexities) if random_perplexities else 0
            },
            "contrast_ratio": avg_random / avg_known if avg_known else 0
        }
