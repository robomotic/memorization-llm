"""
Utilities for interacting with LLM APIs (OpenRouter and Azure OpenAI).
"""
import json
import logging
import time
import requests
from typing import Dict, List, Any, Optional, Union, Literal

from src.config import (
    API_BACKEND,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT
)

logger = logging.getLogger(__name__)

class APIClient:
    """Client for interacting with LLM APIs (OpenRouter and Azure OpenAI)."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                base_url: Optional[str] = None,
                backend: Optional[str] = None):
        """Initialize the API client.
        
        Args:
            api_key: The API key. If None, uses the one from config based on backend.
            base_url: The base URL for the API. If None, uses the one from config.
            backend: The API backend to use ('openrouter' or 'azure'). If None, uses config.
        """
        self.backend = backend.lower() if backend else API_BACKEND
        if self.backend not in ["openrouter", "azure"]:
            raise ValueError(f"Invalid backend: {self.backend}. Must be 'openrouter' or 'azure'.")
        
        # Set up API credentials based on backend
        if self.backend == "openrouter":
            self.api_key = api_key or OPENROUTER_API_KEY
            if not self.api_key:
                raise ValueError("OpenRouter API key is not set. Please set OPENROUTER_API_KEY in .env")
            
            self.base_url = base_url or OPENROUTER_BASE_URL
            self.session = requests.Session()
            self.session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/memorization-llm-detection-tool"
            })
        else:  # Azure OpenAI
            self.api_key = api_key or AZURE_OPENAI_API_KEY
            if not self.api_key:
                raise ValueError("Azure OpenAI API key is not set. Please set AZURE_OPENAI_API_KEY in .env")
            
            self.endpoint = AZURE_OPENAI_ENDPOINT
            if not self.endpoint:
                raise ValueError("Azure OpenAI endpoint is not set. Please set AZURE_OPENAI_ENDPOINT in .env")
            
            self.api_version = AZURE_OPENAI_API_VERSION
            self.deployment = AZURE_OPENAI_DEPLOYMENT
            if not self.deployment:
                raise ValueError("Azure OpenAI deployment is not set. Please set AZURE_OPENAI_DEPLOYMENT in .env")
            
            self.session = requests.Session()
            self.session.headers.update({
                "Content-Type": "application/json",
                "api-key": self.api_key
            })
    
    def get_completion(self, 
                      prompt: str, 
                      model: str,
                      temperature: float = 0.7,
                      max_tokens: int = 150) -> Dict[str, Any]:
        """Get a completion from the LLM API.
        
        Args:
            prompt: The prompt to send to the model.
            model: The model to use (for OpenRouter) or ignored (for Azure).
            temperature: The temperature parameter.
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            The API response as a dictionary.
        """
        if self.backend == "openrouter":
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        else:  # Azure OpenAI
            url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling {self.backend.capitalize()} API: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response content: {e.response.text}")
            raise
    
    def get_token_probabilities(self,
                               prompt: str,
                               model: str,
                               temperature: float = 0.0,
                               max_tokens: int = 1,
                               logprobs: int = 5) -> Dict[str, Any]:
        """Get token probabilities from the LLM API.
        
        Args:
            prompt: The prompt to send to the model.
            model: The model to use (for OpenRouter) or ignored (for Azure).
            temperature: The temperature parameter (usually 0 for probabilities).
            max_tokens: The maximum number of tokens to generate.
            logprobs: The number of most likely tokens to return probabilities for.
            
        Returns:
            The API response with token probabilities.
        """
        if self.backend == "openrouter":
            url = f"{self.base_url}/completions"  # Use completions endpoint for logprobs
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "logprobs": logprobs,
                "echo": True  # Return the prompt with the response
            }
        else:  # Azure OpenAI
            url = f"{self.endpoint}/openai/deployments/{self.deployment}/completions?api-version={self.api_version}"
            payload = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "logprobs": logprobs,
                "echo": True  # Return the prompt with the response
            }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling {self.backend.capitalize()} API for token probabilities: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response content: {e.response.text}")
            
            # If the API doesn't support logprobs, provide a helpful error message
            if hasattr(e, 'response') and e.response and e.response.status_code == 400:
                if "logprobs" in e.response.text.lower():
                    logger.error("This API endpoint may not support logprobs. Consider using a different backend or model.")
            raise
            
    def calculate_sequence_probability(self, 
                                     prompt: str, 
                                     continuation: str, 
                                     model: str) -> float:
        """Calculate the probability of a continuation given a prompt.
        
        Args:
            prompt: The prompt context.
            continuation: The continuation to calculate probability for.
            model: The model to use (for OpenRouter) or ignored (for Azure).
            
        Returns:
            The log probability of the continuation.
        """
        # For models that support it, we'll get token-by-token probabilities
        full_text = prompt + continuation
        
        try:
            # Get token probabilities from the API
            response = self.get_token_probabilities(full_text, model)
            
            # Extract token probabilities for the continuation portion
            # This is a simplified approach - real implementation would need to align
            # tokenization boundaries correctly
            tokens = response.get('choices', [{}])[0].get('logprobs', {}).get('tokens', [])
            token_logprobs = response.get('choices', [{}])[0].get('logprobs', {}).get('token_logprobs', [])
            
            # Find where the continuation starts in the tokenized sequence
            # This is approximate and would need to be refined based on the tokenizer
            prompt_length = len(prompt)
            total_logprob = 0.0
            token_count = 0
            
            # Sum log probabilities for tokens in the continuation
            current_length = 0
            for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
                if logprob is None:  # Skip the first token which might have None logprob
                    continue
                current_length += len(token)
                if current_length > prompt_length:
                    total_logprob += logprob
                    token_count += 1
            
            # Return average log probability if we have tokens
            if token_count > 0:
                return total_logprob / token_count
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating sequence probability: {e}")
            # Fallback approach if logprobs are not supported
            logger.warning("Falling back to approximate probability calculation")
            
            # Use a simple completion-based approach as fallback
            # This is less accurate but works with models that don't support logprobs
            try:
                # Get completion with the prompt
                response = self.get_completion(
                    prompt=prompt,
                    model=model,
                    temperature=0.0,  # Use deterministic output
                    max_tokens=len(continuation.split()) * 2  # Estimate token count
                )
                
                # Extract generated text
                generated = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Compare with expected continuation
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, continuation.lower(), generated.lower()).ratio()
                
                # Convert similarity to a log probability-like value
                # This is a very rough approximation
                import math
                approx_logprob = math.log(max(similarity, 0.01))
                
                return approx_logprob
            except Exception as fallback_error:
                logger.error(f"Fallback probability calculation failed: {fallback_error}")
                return -10.0  # Return a low probability as default
