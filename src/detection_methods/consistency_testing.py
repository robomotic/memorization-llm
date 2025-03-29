"""
Consistency testing methods for memorization detection in multiple-choice settings.
"""
import logging
import random
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from src.utils.api import APIClient
from src.utils.data_loader import format_modified_prompt
from src.config import CONSISTENCY_TRIALS

logger = logging.getLogger(__name__)

class ConsistencyTestingDetector:
    """Detector using consistency testing for memorization detection."""
    
    def __init__(self, 
               model: str, 
               api_client: Optional[APIClient] = None,
               num_trials: Optional[int] = None):
        """Initialize the consistency testing detector.
        
        Args:
            model: Model identifier to use for detection.
            api_client: Optional API client. If None, a new one is created.
            num_trials: Number of consistency trials to run. If None, uses config.
        """
        self.model = model
        self.api_client = api_client or APIClient()
        self.num_trials = num_trials or CONSISTENCY_TRIALS
    
    def _get_model_answer(self, prompt: str) -> str:
        """Get model's answer for a prompt.
        
        Args:
            prompt: The formatted prompt to send to the model.
            
        Returns:
            Model's answer (just the letter).
        """
        # Get completion from model
        response = self.api_client.get_completion(
            prompt=prompt,
            model=self.model,
            temperature=0.1,  # Lower temperature for more consistent answers
            max_tokens=50
        )
        
        # Extract answer text
        answer_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract just the answer letter (A, B, C, or D)
        import re
        match = re.search(r'([A-D])[.\s]', answer_text)
        if match:
            return match.group(1)
        
        # If no match found, try to find any single letter
        match = re.search(r'[A-D]', answer_text)
        if match:
            return match.group(0)
        
        # If still no match, return empty string
        return ""
    
    def detect(self, question: str, options: List[str], correct_idx: int) -> Dict[str, Any]:
        """Detect memorization using consistency testing.
        
        Args:
            question: The question text.
            options: List of answer options.
            correct_idx: Index of the correct answer.
            
        Returns:
            Dictionary with detection results.
        """
        logger.info(f"Running consistency testing detection for question: {question[:50]}...")
        
        # First, get the model's answer to the original prompt
        original_prompt = f"Question: {question}\n\nOptions:\n"
        for i, option in enumerate(options):
            option_letter = chr(65 + i)  # A, B, C, D
            original_prompt += f"{option_letter}. {option}\n"
        original_prompt += "\nAnswer:"
        
        original_answer = self._get_model_answer(original_prompt)
        
        # Convert original answer to index
        if original_answer:
            original_idx = ord(original_answer) - ord('A')
        else:
            # If no clear answer, use a random index (will lead to low consistency)
            original_idx = random.randint(0, len(options) - 1)
        
        # Run multiple trials with modified prompts
        modification_types = ["reordered", "rephrased", "both"]
        trial_results = []
        
        for _ in range(self.num_trials):
            # Select a random modification type
            mod_type = random.choice(modification_types)
            
            # Create modified prompt
            modified_prompt, reordered_options = format_modified_prompt(
                question=question,
                options=options,
                modification_type=mod_type
            )
            
            # Get model's answer to modified prompt
            modified_answer = self._get_model_answer(modified_prompt)
            
            # Convert modified answer to original option index
            if modified_answer:
                modified_letter_idx = ord(modified_answer) - ord('A')
                
                # If options were reordered, map back to original indices
                if mod_type in ["reordered", "both"]:
                    # Find where each original option is in the reordered list
                    option_mapping = {}
                    for i, opt in enumerate(reordered_options):
                        for j, orig_opt in enumerate(options):
                            if opt == orig_opt:
                                option_mapping[i] = j
                    
                    # Map the modified answer index to original index
                    if modified_letter_idx in option_mapping:
                        modified_idx = option_mapping[modified_letter_idx]
                    else:
                        modified_idx = -1
                else:
                    # If only rephrased, index is the same
                    modified_idx = modified_letter_idx
            else:
                # If no clear answer, use a value that won't match
                modified_idx = -1
            
            # Check if modified answer matches original
            is_consistent = modified_idx == original_idx
            
            # Add trial result
            trial_results.append({
                "modification_type": mod_type,
                "is_consistent": is_consistent,
                "original_idx": original_idx,
                "modified_idx": modified_idx
            })
        
        # Calculate consistency score
        consistent_count = sum(1 for trial in trial_results if trial["is_consistent"])
        consistency_score = consistent_count / len(trial_results) if trial_results else 0
        
        # Check if original answer was correct
        original_correct = original_idx == correct_idx
        
        # Calculate memorization score
        # Lower consistency with correct original answer suggests memorization
        if original_correct:
            # If original answer was correct but consistency is low,
            # the model might be memorizing instead of understanding
            memorization_score = 1.0 - consistency_score
        else:
            # If original answer was wrong, low consistency doesn't indicate memorization
            memorization_score = 0.0
        
        # Determine if memorization is detected
        # Threshold: If original answer correct but consistency < 0.7,
        # consider it potential memorization
        is_memorized = original_correct and consistency_score < 0.7
        
        return {
            "method": "consistency_testing",
            "memorization_score": float(memorization_score),
            "is_memorized": bool(is_memorized),
            "consistency_score": float(consistency_score),
            "original_correct": bool(original_correct),
            "original_answer": original_answer,
            "original_idx": int(original_idx),
            "trial_results": trial_results,
            "threshold": 0.7
        }
    
    def detect_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run consistency testing detection on a batch of questions.
        
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
