"""
Perplexity-based methods for memorization detection in multiple-choice settings.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from src.utils.api import OpenRouterClient
from src.config import PERPLEXITY_THRESHOLD

logger = logging.getLogger(__name__)

class PerplexityDetector:
    """Detector using perplexity-based methods for multiple-choice questions."""
    
    def __init__(self, model: str, api_client: Optional[OpenRouterClient] = None):
        """Initialize the perplexity detector.
        
        Args:
            model: Model identifier to use for detection.
            api_client: Optional API client. If None, a new one is created.
        """
        self.model = model
        self.api_client = api_client or OpenRouterClient()
    
    def detect(self, question: str, options: List[str], correct_idx: int) -> Dict[str, Any]:
        """Detect memorization using perplexity-based methods.
        
        Args:
            question: The question text.
            options: List of answer options.
            correct_idx: Index of the correct answer.
            
        Returns:
            Dictionary with detection results.
        """
        logger.info(f"Running perplexity-based detection for question: {question[:50]}...")
        
        # Calculate perplexity for each option
        option_probs = []
        for option in options:
            # Format prompt for probability calculation
            prompt = f"Question: {question}\nAnswer: {option}"
            
            # Calculate probability
            logprob = self.api_client.calculate_sequence_probability(
                prompt=question,
                continuation=option,
                model=self.model
            )
            option_probs.append(logprob)
        
        # Extract correct option probability and average of incorrect options
        correct_prob = option_probs[correct_idx]
        incorrect_probs = [p for i, p in enumerate(option_probs) if i != correct_idx]
        avg_incorrect = sum(incorrect_probs) / len(incorrect_probs) if incorrect_probs else 0
        
        # Calculate memorization score
        if avg_incorrect == 0:
            memorization_score = 0.0
        else:
            # Using negative logprobs, so larger values are less likely
            # Calculate ratio between correct and incorrect (on probability scale)
            memorization_score = np.exp(correct_prob) / np.exp(avg_incorrect)
        
        # Determine if memorization is detected based on threshold
        is_memorized = memorization_score > PERPLEXITY_THRESHOLD
        
        return {
            "method": "perplexity",
            "memorization_score": float(memorization_score),
            "is_memorized": bool(is_memorized),
            "option_probs": option_probs,
            "correct_prob": float(correct_prob),
            "avg_incorrect_prob": float(avg_incorrect),
            "threshold": PERPLEXITY_THRESHOLD
        }
        
    def detect_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run perplexity detection on a batch of questions.
        
        Args:
            batch: List of preprocessed examples.
            
        Returns:
            List of detection results for each example.
        """
        results = []
        for example in batch:
            result = self.detect(
                question=example["question"],
                options=example["options"],
                correct_idx=example["correct_idx"]
            )
            # Add example ID to result
            result["id"] = example.get("id", "")
            results.append(result)
        
        return results
