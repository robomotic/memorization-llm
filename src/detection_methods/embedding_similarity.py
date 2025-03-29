"""
Embedding similarity methods for memorization detection in multiple-choice settings.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.api import APIClient
from src.config import EMBEDDING_MODEL, EMBEDDING_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class EmbeddingSimilarityDetector:
    """Detector using embedding similarity for memorization detection."""
    
    def __init__(self, 
               model: str, 
               api_client: Optional[APIClient] = None,
               embedding_model: Optional[str] = None):
        """Initialize the embedding similarity detector.
        
        Args:
            model: LLM identifier to use for generating completions.
            api_client: Optional API client. If None, a new one is created.
            embedding_model: Model to use for embeddings. If None, uses config value.
        """
        self.model = model
        self.api_client = api_client or APIClient()
        
        # Load the embedding model
        embedding_model_name = embedding_model or EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
    
    def _get_model_completion(self, question: str, options: List[str]) -> str:
        """Get model completion for a multiple-choice question.
        
        Args:
            question: The question text.
            options: List of answer options.
            
        Returns:
            Model's completion text.
        """
        # Format options
        formatted_options = ""
        for i, option in enumerate(options):
            option_letter = chr(65 + i)  # A, B, C, D
            formatted_options += f"{option_letter}. {option}\n"
        
        # Format prompt
        prompt = f"""Question: {question}

Options:
{formatted_options}

I'll solve this medical question step by step and provide the correct answer with detailed reasoning."""
        
        # Get completion from model
        response = self.api_client.get_completion(
            prompt=prompt,
            model=self.model,
            temperature=0.7,
            max_tokens=300
        )
        
        # Extract completion text
        completion = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return completion
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using embeddings.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Cosine similarity score (0-1).
        """
        # Generate embeddings
        embedding1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
        return cosine_similarity.item()
    
    def detect(self, question: str, options: List[str], 
              reference_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect memorization using embedding similarity.
        
        Args:
            question: The question text.
            options: List of answer options.
            reference_texts: Optional list of reference texts to compare against.
                If None, uses options as references.
            
        Returns:
            Dictionary with detection results.
        """
        logger.info(f"Running embedding similarity detection for question: {question[:50]}...")
        
        # Get model completion
        model_completion = self._get_model_completion(question, options)
        
        # If no reference texts provided, use options
        if not reference_texts:
            reference_texts = options
        
        # Calculate similarity with each reference text
        similarities = []
        for ref_text in reference_texts:
            similarity = self._compute_similarity(model_completion, ref_text)
            similarities.append(similarity)
        
        # Get maximum similarity across references
        max_similarity = max(similarities) if similarities else 0.0
        
        # Determine if memorization is detected based on threshold
        is_memorized = max_similarity > EMBEDDING_SIMILARITY_THRESHOLD
        
        return {
            "method": "embedding_similarity",
            "memorization_score": float(max_similarity),
            "is_memorized": bool(is_memorized),
            "similarities": [float(s) for s in similarities],
            "model_completion": model_completion,
            "threshold": EMBEDDING_SIMILARITY_THRESHOLD
        }
    
    def detect_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run embedding similarity detection on a batch of questions.
        
        Args:
            batch: List of preprocessed examples.
            
        Returns:
            List of detection results for each example.
        """
        results = []
        for example in batch:
            # Use both question and options as reference texts
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
