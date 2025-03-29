"""
N-gram overlap methods for memorization detection in multiple-choice settings.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter

import nltk
from nltk.util import ngrams

from src.utils.api import APIClient
from src.config import NGRAM_SIZES

# Download nltk resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

logger = logging.getLogger(__name__)

class NgramOverlapDetector:
    """Detector using n-gram overlap for memorization detection."""
    
    def __init__(self, 
               model: str, 
               api_client: Optional[APIClient] = None,
               ngram_sizes: Optional[List[int]] = None):
        """Initialize the n-gram overlap detector.
        
        Args:
            model: Model identifier to use for detection.
            api_client: Optional API client. If None, a new one is created.
            ngram_sizes: Sizes of n-grams to analyze. If None, uses config values.
        """
        self.model = model
        self.api_client = api_client or APIClient()
        self.ngram_sizes = ngram_sizes or NGRAM_SIZES
    
    def _extract_ngrams(self, text: str, n: int) -> Set[Tuple[str, ...]]:
        """Extract n-grams from text.
        
        Args:
            text: Input text.
            n: Size of n-grams.
            
        Returns:
            Set of n-grams.
        """
        tokens = nltk.word_tokenize(text.lower())
        return set(ngrams(tokens, n))
    
    def _get_model_explanation(self, question: str, options: List[str]) -> str:
        """Get model explanation for a multiple-choice question.
        
        Args:
            question: The question text.
            options: List of answer options.
            
        Returns:
            Model's explanation text.
        """
        # Format options
        formatted_options = ""
        for i, option in enumerate(options):
            option_letter = chr(65 + i)  # A, B, C, D
            formatted_options += f"{option_letter}. {option}\n"
        
        # Format prompt to ask for explanation
        prompt = f"""Question: {question}

Options:
{formatted_options}

Please provide a detailed explanation for the correct answer to this medical question. 
Include your reasoning and relevant medical knowledge."""
        
        # Get completion from model
        response = self.api_client.get_completion(
            prompt=prompt,
            model=self.model,
            temperature=0.7,
            max_tokens=300
        )
        
        # Extract explanation text
        explanation = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return explanation
    
    def detect(self, question: str, options: List[str], 
              reference_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect memorization using n-gram overlap.
        
        Args:
            question: The question text.
            options: List of answer options.
            reference_texts: Optional list of reference texts to compare against.
                If None, uses the question and options as references.
            
        Returns:
            Dictionary with detection results.
        """
        logger.info(f"Running n-gram overlap detection for question: {question[:50]}...")
        
        # Get model explanation
        model_explanation = self._get_model_explanation(question, options)
        
        # If no reference texts provided, use question and options
        if not reference_texts:
            reference_texts = [question] + options
        
        # Initialize results for each n-gram size
        overlap_results = {}
        max_overlap_ratio = 0.0
        
        # Calculate overlap for each n-gram size
        for n in self.ngram_sizes:
            # Extract n-grams from model explanation
            explanation_ngrams = self._extract_ngrams(model_explanation, n)
            
            # Calculate overlap with each reference text
            reference_overlaps = []
            for ref_text in reference_texts:
                ref_ngrams = self._extract_ngrams(ref_text, n)
                
                # Skip if no n-grams in reference (e.g., text too short)
                if not ref_ngrams:
                    continue
                
                # Calculate overlap ratio
                overlap = explanation_ngrams.intersection(ref_ngrams)
                overlap_ratio = len(overlap) / len(ref_ngrams) if ref_ngrams else 0
                
                reference_overlaps.append({
                    "overlap_count": len(overlap),
                    "ref_count": len(ref_ngrams),
                    "overlap_ratio": overlap_ratio
                })
            
            # Get maximum overlap ratio across references
            max_ref_overlap = max(
                [overlap["overlap_ratio"] for overlap in reference_overlaps],
                default=0.0
            )
            
            # Update max overlap ratio across all n-gram sizes
            max_overlap_ratio = max(max_overlap_ratio, max_ref_overlap)
            
            # Store results for this n-gram size
            overlap_results[f"{n}-gram"] = {
                "explanation_count": len(explanation_ngrams),
                "reference_overlaps": reference_overlaps,
                "max_overlap_ratio": max_ref_overlap
            }
        
        # Determine if memorization is detected based on threshold
        # Heuristic: If any n-gram size has overlap ratio > 0.3, consider memorized
        is_memorized = max_overlap_ratio > 0.3
        
        return {
            "method": "ngram_overlap",
            "memorization_score": float(max_overlap_ratio),
            "is_memorized": bool(is_memorized),
            "ngram_results": overlap_results,
            "model_explanation": model_explanation,
            "threshold": 0.3
        }
    
    def detect_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run n-gram overlap detection on a batch of questions.
        
        Args:
            batch: List of preprocessed examples.
            
        Returns:
            List of detection results for each example.
        """
        results = []
        for example in batch:
            # Combine question and options as reference texts
            reference_texts = [example["question"]] + example["options"]
            
            result = self.detect(
                question=example["question"],
                options=example["options"],
                reference_texts=reference_texts
            )
            
            # Add example ID to result
            result["id"] = example.get("id", "")
            results.append(result)
        
        return results
