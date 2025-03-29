"""
Unit tests for perplexity-based memorization detection.
Tests perplexity calculations on various types of content:
1. Random high entropy strings
2. Known facts (likely in training data)
3. Progressive token-level perplexity calculation
"""
import unittest
import json
import random
import string
import numpy as np
from .perplexity_utils import PerplexityAnalyzer

class TestPerplexity(unittest.TestCase):
    """Tests for perplexity-based memorization detection."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = PerplexityAnalyzer(model_name="gpt-3.5-turbo")
        
        # Sample texts of different types
        self.known_facts = [
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The Earth revolves around the Sun.",
            "The speed of light is approximately 299,792,458 meters per second.",
            "The Declaration of Independence was signed in 1776.",
        ]
        
        # Generate random high-entropy strings
        self.random_strings = []
        for _ in range(5):
            # Generate random string of letters, numbers, and special characters
            random_string = ''.join(random.choices(
                string.ascii_letters + string.digits + string.punctuation, 
                k=random.randint(50, 100)
            ))
            self.random_strings.append(random_string)
    
    def test_tokenization(self):
        """Test that text tokenization works properly."""
        for text in self.known_facts:
            tokens = self.analyzer.tokenize(text)
            decoded_tokens = self.analyzer.decode_tokens(tokens)
            
            # Verify tokens can be decoded back to something close to original
            # (exact reconstruction may not be possible due to tokenization artifacts)
            reconstructed = ''.join(decoded_tokens)
            self.assertEqual(text, reconstructed)
    
    def test_perplexity_of_known_facts(self):
        """Test perplexity calculation on known facts (likely in training data)."""
        results = []
        
        print("\n=== Perplexity of Known Facts ===")
        for fact in self.known_facts:
            result = self.analyzer.calculate_token_logprobs(fact)
            
            print(f"Fact: {fact}")
            print(f"Perplexity: {result['perplexity']:.4f}")
            print(f"Avg Log Probability: {result['avg_logprob']:.4f}")
            print("-" * 80)
            
            results.append({
                'text': fact,
                'perplexity': result['perplexity'],
                'avg_logprob': result['avg_logprob']
            })
        
        # Save results to file for further analysis
        with open('fact_perplexity_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Known facts should generally have lower perplexity (higher predictability)
        for result in results:
            self.assertLess(result['perplexity'], 100, 
                            f"Known fact has unexpectedly high perplexity: {result['text']}")
    
    def test_perplexity_of_random_strings(self):
        """Test perplexity calculation on random high-entropy strings."""
        results = []
        
        print("\n=== Perplexity of Random Strings ===")
        for random_string in self.random_strings:
            result = self.analyzer.calculate_token_logprobs(random_string)
            
            preview = random_string[:30] + "..." if len(random_string) > 30 else random_string
            print(f"Random string: {preview}")
            print(f"Perplexity: {result['perplexity']:.4f}")
            print(f"Avg Log Probability: {result['avg_logprob']:.4f}")
            print("-" * 80)
            
            results.append({
                'text': preview,
                'perplexity': result['perplexity'],
                'avg_logprob': result['avg_logprob']
            })
        
        # Save results to file for further analysis
        with open('random_perplexity_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Random strings should generally have higher perplexity (lower predictability)
        for result in results:
            self.assertGreater(result['perplexity'], 10, 
                               f"Random string has unexpectedly low perplexity: {result['text']}")
    
    def test_comparison_between_types(self):
        """Compare perplexity between known facts and random strings."""
        # Calculate average perplexity for each category
        fact_perplexities = []
        for fact in self.known_facts:
            result = self.analyzer.calculate_token_logprobs(fact)
            fact_perplexities.append(result['perplexity'])
        
        random_perplexities = []
        for random_string in self.random_strings:
            result = self.analyzer.calculate_token_logprobs(random_string)
            random_perplexities.append(result['perplexity'])
        
        avg_fact_perplexity = sum(fact_perplexities) / len(fact_perplexities)
        avg_random_perplexity = sum(random_perplexities) / len(random_perplexities)
        
        print("\n=== Perplexity Comparison ===")
        print(f"Average Known Fact Perplexity: {avg_fact_perplexity:.4f}")
        print(f"Average Random String Perplexity: {avg_random_perplexity:.4f}")
        print(f"Ratio (Random/Fact): {avg_random_perplexity/avg_fact_perplexity:.4f}")
        
        # Random strings should have significantly higher perplexity than known facts
        self.assertGreater(avg_random_perplexity, avg_fact_perplexity,
                         "Random strings should have higher perplexity than known facts")
    
    def test_token_level_perplexity(self):
        """Test progressive token-level perplexity calculation."""
        # Choose one known fact for detailed analysis
        fact = "The capital of France is Paris."
        
        print("\n=== Progressive Token-Level Perplexity ===")
        print(f"Analyzing: '{fact}'")
        
        # Calculate perplexity at each token position
        results = self.analyzer.calculate_token_level_perplexity(fact)
        
        print("Position\tSuffix\t\t\t\tPerplexity\tAvg LogProb")
        print("-" * 80)
        
        for result in results:
            # Format prefix for display (truncate if too long)
            prefix = result['prefix_text']
            if len(prefix) > 30:
                prefix = prefix[:27] + "..."
            
            print(f"{result['position']}\t{prefix.ljust(30)}\t{result['perplexity']:.4f}\t{result['avg_logprob']:.4f}")
        
        # Save token-level results to file
        with open('token_level_perplexity.json', 'w') as f:
            # Convert NumPy values to native Python types for JSON serialization
            serializable_results = []
            for result in results:
                serializable_result = {k: v for k, v in result.items()}
                serializable_result['avg_logprob'] = float(result['avg_logprob'])
                serializable_result['perplexity'] = float(result['perplexity'])
                serializable_result['logprobs'] = [float(lp) for lp in result['logprobs']]
                serializable_results.append(serializable_result)
                
            json.dump(serializable_results, f, indent=2)
        
        # The most specific part (e.g., "Paris") should have higher perplexity
        # when considered alone than the full sentence
        paris_idx = len(results) - 1  # Last token position
        full_sentence_idx = 0  # First position (full sentence)
        
        self.assertGreater(results[paris_idx]['perplexity'], results[full_sentence_idx]['perplexity'],
                         "Context-specific tokens should have higher perplexity than full sentence")

if __name__ == '__main__':
    unittest.main()
