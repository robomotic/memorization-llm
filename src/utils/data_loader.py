"""
Utilities for loading and processing the MedQA-USMLE dataset.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Tuple

import datasets
import pandas as pd

from src.config import MEDQA_DATASET_NAME, CACHE_DIR

logger = logging.getLogger(__name__)

def load_medqa_dataset(split: str = "train") -> datasets.Dataset:
    """Load the MedQA-USMLE dataset.
    
    Args:
        split: The dataset split to load (train, validation, or test).
        
    Returns:
        The loaded dataset.
    """
    logger.info(f"Loading MedQA-USMLE dataset (split: {split})...")
    
    try:
        dataset = datasets.load_dataset(MEDQA_DATASET_NAME, split=split, cache_dir=CACHE_DIR)
        logger.info(f"Loaded {len(dataset)} examples from MedQA-USMLE ({split} split)")
        return dataset
    except Exception as e:
        logger.error(f"Error loading MedQA-USMLE dataset: {e}")
        raise

def format_multiple_choice_prompt(question: str, options: List[str]) -> str:
    """Format a multiple-choice question for prompting LLMs.
    
    Args:
        question: The question text.
        options: List of answer options.
        
    Returns:
        Formatted prompt string.
    """
    formatted_prompt = f"Question: {question}\n\nOptions:\n"
    
    for i, option in enumerate(options):
        option_letter = chr(65 + i)  # A, B, C, D
        formatted_prompt += f"{option_letter}. {option}\n"
    
    formatted_prompt += "\nAnswer:"
    
    return formatted_prompt

def format_modified_prompt(question: str, options: List[str], 
                         modification_type: str = "reordered") -> Tuple[str, List[str]]:
    """Create a modified version of the prompt for consistency testing.
    
    Args:
        question: The question text.
        options: List of answer options.
        modification_type: Type of modification (reordered, rephrased, or both).
        
    Returns:
        Tuple of (modified prompt, reordered options list)
    """
    import random
    from nltk.tokenize import word_tokenize
    
    # Create a reordered version of the options
    reordered_indices = list(range(len(options)))
    while reordered_indices == list(range(len(options))):
        random.shuffle(reordered_indices)
    
    reordered_options = [options[i] for i in reordered_indices]
    
    # For reordered only, just change the options order
    if modification_type == "reordered":
        prompt = format_multiple_choice_prompt(question, reordered_options)
        return prompt, reordered_options
    
    # For rephrased, modify the question while keeping the meaning
    elif modification_type == "rephrased":
        # Simple rephrasing by adding introductory phrases
        intro_phrases = [
            "Based on medical knowledge, ",
            "According to clinical guidelines, ",
            "From a diagnostic perspective, ",
            "In clinical practice, ",
            "As per medical literature, "
        ]
        
        rephrased_question = random.choice(intro_phrases) + question.lower()
        prompt = format_multiple_choice_prompt(rephrased_question, options)
        return prompt, options
    
    # For both, combine the modifications
    else:
        intro_phrases = [
            "Based on medical knowledge, ",
            "According to clinical guidelines, ",
            "From a diagnostic perspective, ",
            "In clinical practice, ",
            "As per medical literature, "
        ]
        
        rephrased_question = random.choice(intro_phrases) + question.lower()
        prompt = format_multiple_choice_prompt(rephrased_question, reordered_options)
        return prompt, reordered_options

def get_answer_index(options: List[str], correct_answer: str) -> int:
    """Get the index of the correct answer in the options list.
    
    Args:
        options: List of answer options.
        correct_answer: The correct answer text.
        
    Returns:
        Index of the correct answer.
    """
    for i, option in enumerate(options):
        if option.strip() == correct_answer.strip():
            return i
    
    # If no exact match, try case-insensitive and partial matching
    for i, option in enumerate(options):
        if option.lower().strip() == correct_answer.lower().strip():
            return i
        
    # If still no match, find the closest option
    import difflib
    matches = difflib.get_close_matches(correct_answer, options, n=1, cutoff=0.6)
    if matches:
        return options.index(matches[0])
    
    # If all else fails, return -1
    logger.warning(f"Could not find correct answer '{correct_answer}' in options: {options}")
    return -1

def preprocess_medqa_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a MedQA example to standardize format.
    
    Args:
        example: A single example from the MedQA dataset.
        
    Returns:
        Preprocessed example with standardized fields.
    """
    # Extract options
    options = [
        example.get('options', {}).get('A', ''),
        example.get('options', {}).get('B', ''),
        example.get('options', {}).get('C', ''),
        example.get('options', {}).get('D', '')
    ]
    
    # Get the question text
    question = example.get('question', '')
    
    # Get the correct answer
    correct_answer = example.get('answer', '')
    correct_idx = ord(correct_answer) - ord('A') if len(correct_answer) == 1 else -1
    
    # Format the standard prompt
    formatted_prompt = format_multiple_choice_prompt(question, options)
    
    return {
        'id': example.get('id', ''),
        'question': question,
        'options': options,
        'formatted_prompt': formatted_prompt,
        'correct_answer': correct_answer,
        'correct_idx': correct_idx
    }
