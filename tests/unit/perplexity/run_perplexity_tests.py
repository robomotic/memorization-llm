#!/usr/bin/env python3
"""
Perplexity test runner script.
Runs perplexity tests with various strings and analyzes the results.
"""
import os
import sys
import json
import random
import string
import logging
import argparse
import time
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv

# Add the project root to Python path for imports to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from perplexity_utils import PerplexityAnalyzer
from api_perplexity import APIPerplexityAnalyzer

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('perplexity_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_random_string(length: int) -> str:
    """Generate a random string with high entropy.
    
    Args:
        length: Length of the string to generate
        
    Returns:
        Random string
    """
    return ''.join(random.choices(
        string.ascii_letters + string.digits + string.punctuation, 
        k=length
    ))

def generate_test_data():
    """Generate test data for perplexity analysis.
    
    Returns:
        Dictionary with test data categories
    """
    # Known facts likely to be in training data
    known_facts = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth revolves around the Sun.",
        "The speed of light is approximately 299,792,458 meters per second.",
        "The Declaration of Independence was signed in 1776.",
        "Mount Everest is the tallest mountain on Earth.",
        "The human body is composed of approximately 60% water.",
        "William Shakespeare wrote Romeo and Juliet.",
        "The Great Wall of China is visible from space.",
        "The mitochondria is the powerhouse of the cell."
    ]
    
    # Generate random high-entropy strings
    random_strings = [generate_random_string(random.randint(50, 100)) for _ in range(10)]
    
    # Sentences with increasing complexity
    complexity_progression = [
        "The cat sat on the mat.",
        "The large black cat sat on the worn-out doormat.",
        "The neighbor's elderly black cat with a white spot on its paw sat lazily on our colorful doormat.",
        "The neighbor's particularly finicky seventeen-year-old black cat with unusual white markings on its front paws sat contentedly on our recently purchased handmade Moroccan doormat."
    ]
    
    # Sentences with increasing specificity (easy to hard to predict)
    specificity_progression = [
        "The country has a capital.",
        "The European country has a capital.",
        "France has a capital.",
        "The capital of France is Paris."
    ]
    
    return {
        "known_facts": known_facts,
        "random_strings": random_strings,
        "complexity_progression": complexity_progression,
        "specificity_progression": specificity_progression
    }

def run_mock_tests():
    """Run perplexity tests using the mock analyzer."""
    logger.info("Running mock perplexity tests...")
    
    # Generate test data
    test_data = generate_test_data()
    
    # Initialize analyzer
    analyzer = PerplexityAnalyzer(model_name="gpt-3.5-turbo")
    
    # Test known facts
    fact_results = []
    logger.info("Testing known facts...")
    for fact in test_data["known_facts"]:
        result = analyzer.calculate_token_logprobs(fact)
        fact_results.append({
            "text": fact,
            "perplexity": result["perplexity"],
            "avg_logprob": result["avg_logprob"],
            "token_count": len(result["tokens"])
        })
        logger.info(f"Fact: '{fact}' - Perplexity: {result['perplexity']:.4f}")
    
    # Test random strings
    random_results = []
    logger.info("Testing random strings...")
    for random_string in test_data["random_strings"]:
        result = analyzer.calculate_token_logprobs(random_string)
        preview = random_string[:30] + "..." if len(random_string) > 30 else random_string
        random_results.append({
            "text": preview,
            "perplexity": result["perplexity"],
            "avg_logprob": result["avg_logprob"],
            "token_count": len(result["tokens"])
        })
        logger.info(f"Random string: '{preview}' - Perplexity: {result['perplexity']:.4f}")
    
    # Test complexity progression
    complexity_results = []
    logger.info("Testing complexity progression...")
    for sentence in test_data["complexity_progression"]:
        result = analyzer.calculate_token_logprobs(sentence)
        complexity_results.append({
            "text": sentence,
            "perplexity": result["perplexity"],
            "avg_logprob": result["avg_logprob"],
            "token_count": len(result["tokens"])
        })
        logger.info(f"Complexity: '{sentence}' - Perplexity: {result['perplexity']:.4f}")
    
    # Test specificity progression
    specificity_results = []
    logger.info("Testing specificity progression...")
    for sentence in test_data["specificity_progression"]:
        result = analyzer.calculate_token_logprobs(sentence)
        specificity_results.append({
            "text": sentence,
            "perplexity": result["perplexity"],
            "avg_logprob": result["avg_logprob"],
            "token_count": len(result["tokens"])
        })
        logger.info(f"Specificity: '{sentence}' - Perplexity: {result['perplexity']:.4f}")
    
    # Test token-level perplexity for a specific fact
    token_level_results = analyzer.calculate_token_level_perplexity("The capital of France is Paris.")
    logger.info("Token-level perplexity for 'The capital of France is Paris.':")
    for result in token_level_results:
        logger.info(f"Position {result['position']}: '{result['prefix_text']}' - Perplexity: {result['perplexity']:.4f}")
    
    # Save all results to file
    results = {
        "known_facts": fact_results,
        "random_strings": random_results,
        "complexity_progression": complexity_results,
        "specificity_progression": specificity_results,
        "token_level_results": token_level_results
    }
    
    with open('mock_perplexity_results.json', 'w') as f:
        # Convert any non-serializable objects to native Python types
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else x)
    
    logger.info("Mock perplexity tests completed. Results saved to mock_perplexity_results.json")

def run_api_tests(api_backend="openai", api_key=None, model_name="gpt-3.5-turbo"):
    """Run perplexity tests using the real API analyzer.
    
    Args:
        api_backend: API backend to use (openai, azure, or openrouter)
        api_key: Optional API key to use
        model_name: Model name to use for testing
    """
    logger.info(f"Running API perplexity tests using {api_backend} backend...")
    
    # Get API key based on the selected backend
    if not api_key:
        if api_backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("No OpenAI API key found. Set OPENAI_API_KEY in .env file.")
                return
        elif api_backend == "azure":
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                logger.error("No Azure OpenAI API key found. Set AZURE_OPENAI_API_KEY in .env file.")
                return
        elif api_backend == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.error("No OpenRouter API key found. Set OPENROUTER_API_KEY in .env file.")
                return
        else:
            logger.error(f"Invalid API backend: {api_backend}. Must be 'openai', 'azure', or 'openrouter'.")
            return
    
    # Generate test data (using a smaller subset to limit API usage)
    test_data = generate_test_data()
    known_facts = test_data["known_facts"][:3]  # Limit to 3 known facts
    random_strings = test_data["random_strings"][:3]  # Limit to 3 random strings
    
    # Initialize analyzer with the appropriate backend and model
    analyzer = APIPerplexityAnalyzer(model_name=model_name, api_key=api_key, api_backend=api_backend)
    
    try:
        # Compare perplexity between known facts and random strings
        logger.info("Comparing perplexity between known facts and random strings...")
        comparison = analyzer.analyze_perplexity_contrast(known_facts, random_strings)
        
        # Log the results
        logger.info(f"Average known fact perplexity: {comparison['known_texts']['avg_perplexity']:.4f}")
        logger.info(f"Average random string perplexity: {comparison['random_texts']['avg_perplexity']:.4f}")
        logger.info(f"Contrast ratio (random/known): {comparison['contrast_ratio']:.4f}")
        
        # Test progressive perplexity for one specific fact
        fact = "The capital of France is Paris."
        logger.info(f"Testing progressive perplexity for: '{fact}'")
        progressive_results = analyzer.progressive_perplexity(fact)
        
        # Log the progressive results
        for result in progressive_results:
            logger.info(f"Position {result['position']}: '{result['prefix_text']}' - Perplexity: {result['perplexity']:.4f}")
        
        # Save results to file
        results = {
            "perplexity_comparison": comparison,
            "progressive_perplexity": progressive_results
        }
        
        # Create an absolute path to save results in the tests directory
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        os.makedirs(results_dir, exist_ok=True)
        
        # Add timestamp to filename to avoid overwriting previous results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(results_dir, f'api_perplexity_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            # Convert any non-serializable objects to native Python types
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else x)
        
        # Also save a copy with a standard name for easy access
        standard_file = os.path.join(results_dir, 'api_perplexity_results.json')
        with open(standard_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else x)
        
        logger.info(f"API perplexity tests completed. Results saved to {standard_file}")
        
    except Exception as e:
        logger.error(f"Error running API perplexity tests: {e}")

def main():
    """Main entry point for the perplexity test runner."""
    parser = argparse.ArgumentParser(description="Run perplexity tests on various strings.")
    parser.add_argument("--mode", choices=["mock", "api", "both"], default="mock",
                      help="Test mode: mock (default), api, or both")
    parser.add_argument("--api-backend", choices=["openai", "azure", "openrouter"], default="openai",
                      help="API backend to use: openai (default), azure, or openrouter")
    parser.add_argument("--api-key", type=str, help="API key for API tests (overrides .env)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", 
                      help="Model name to use for API tests (default: gpt-3.5-turbo)")
    
    args = parser.parse_args()
    
    if args.mode in ["mock", "both"]:
        run_mock_tests()
    
    if args.mode in ["api", "both"]:
        run_api_tests(api_backend=args.api_backend, api_key=args.api_key, model_name=args.model)

if __name__ == "__main__":
    main()
