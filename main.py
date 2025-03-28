#!/usr/bin/env python
"""
Main script for running memorization detection on the MedQA-USMLE dataset.
"""
import argparse
import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.config import DEFAULT_MODEL, RESULTS_DIR, API_BACKEND
from src.utils.api import APIClient
from src.utils.data_loader import load_medqa_dataset, preprocess_medqa_example
from src.detection_methods.perplexity import PerplexityDetector
from src.detection_methods.ngram_overlap import NgramOverlapDetector
from src.detection_methods.embedding_similarity import EmbeddingSimilarityDetector
from src.detection_methods.consistency_testing import ConsistencyTestingDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memorization_detection.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run memorization detection on MedQA-USMLE dataset."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model identifier to use (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--api-backend",
        type=str,
        default=API_BACKEND,
        choices=["openrouter", "azure"],
        help=f"API backend to use (default: {API_BACKEND})"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: train)"
    )
    
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to analyze (default: 10)"
    )
    
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["perplexity", "ngram", "embedding", "consistency"],
        choices=["perplexity", "ngram", "embedding", "consistency", "all"],
        help="Detection methods to run (default: all methods)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated based on timestamp)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    return parser.parse_args()

def run_detection_pipeline(examples: List[Dict[str, Any]], 
                         model: str,
                         methods: List[str],
                         api_backend: str) -> List[Dict[str, Any]]:
    """Run the detection pipeline on a list of examples.
    
    Args:
        examples: List of preprocessed examples.
        model: Model identifier to use.
        methods: List of detection methods to run.
        api_backend: API backend to use ('openrouter' or 'azure').
        
    Returns:
        List of results for each example.
    """
    # Create API client (shared across detectors)
    api_client = APIClient(backend=api_backend)
    
    # Initialize detectors based on specified methods
    detectors = {}
    if "all" in methods or "perplexity" in methods:
        detectors["perplexity"] = PerplexityDetector(model=model, api_client=api_client)
    
    if "all" in methods or "ngram" in methods:
        detectors["ngram"] = NgramOverlapDetector(model=model, api_client=api_client)
    
    if "all" in methods or "embedding" in methods:
        detectors["embedding"] = EmbeddingSimilarityDetector(model=model, api_client=api_client)
    
    if "all" in methods or "consistency" in methods:
        detectors["consistency"] = ConsistencyTestingDetector(model=model, api_client=api_client)
    
    # Run detection on each example
    results = []
    for example in tqdm(examples, desc="Running detection"):
        example_results = {
            "id": example["id"],
            "question": example["question"],
            "options": example["options"],
            "correct_answer": example["correct_answer"],
            "detection_results": {}
        }
        
        # Run each detector
        for method_name, detector in detectors.items():
            try:
                method_result = detector.detect(
                    question=example["question"],
                    options=example["options"],
                    correct_idx=example["correct_idx"]
                )
                example_results["detection_results"][method_name] = method_result
            except Exception as e:
                logger.error(f"Error running {method_name} detector on example {example['id']}: {e}")
                example_results["detection_results"][method_name] = {
                    "error": str(e),
                    "method": method_name
                }
        
        # Add aggregated memorization score and verdict
        memorization_scores = [
            result.get("memorization_score", 0.0) 
            for result in example_results["detection_results"].values()
            if isinstance(result, dict) and "memorization_score" in result
        ]
        
        is_memorized_flags = [
            result.get("is_memorized", False)
            for result in example_results["detection_results"].values()
            if isinstance(result, dict) and "is_memorized" in result
        ]
        
        if memorization_scores:
            example_results["aggregate_score"] = sum(memorization_scores) / len(memorization_scores)
        else:
            example_results["aggregate_score"] = 0.0
        
        if is_memorized_flags:
            # Consider memorized if at least half of the methods indicate memorization
            example_results["is_memorized"] = sum(is_memorized_flags) >= len(is_memorized_flags) / 2
        else:
            example_results["is_memorized"] = False
        
        results.append(example_results)
    
    return results

def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load and preprocess dataset
    dataset = load_medqa_dataset(split=args.split)
    
    # Convert to list and preprocess
    examples = []
    for item in dataset:
        try:
            preprocessed = preprocess_medqa_example(item)
            examples.append(preprocessed)
        except Exception as e:
            logger.error(f"Error preprocessing example {item.get('id', 'unknown')}: {e}")
    
    # Sample examples if requested
    if args.num_examples < len(examples):
        examples = random.sample(examples, args.num_examples)
    
    logger.info(f"Running detection on {len(examples)} examples from {args.split} split")
    
    # Run detection pipeline
    results = run_detection_pipeline(
        examples=examples,
        model=args.model,
        methods=args.methods,
        api_backend=args.api_backend
    )
    
    # Generate output filename if not provided
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            RESULTS_DIR, 
            f"medqa_memorization_{args.model.replace('/', '_')}_{timestamp}.json"
        )
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump({
            "metadata": {
                "model": args.model,
                "split": args.split,
                "num_examples": len(examples),
                "methods": args.methods,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    memorized_count = sum(1 for r in results if r.get("is_memorized", False))
    logger.info(f"Summary: {memorized_count}/{len(results)} examples detected as memorized")
    
    # Print example of memorized and non-memorized questions
    memorized_examples = [r for r in results if r.get("is_memorized", False)]
    non_memorized_examples = [r for r in results if not r.get("is_memorized", False)]
    
    if memorized_examples:
        example = memorized_examples[0]
        logger.info("\nExample of memorized question:")
        logger.info(f"Question: {example['question']}")
        logger.info(f"Aggregate Score: {example['aggregate_score']:.4f}")
    
    if non_memorized_examples:
        example = non_memorized_examples[0]
        logger.info("\nExample of non-memorized question:")
        logger.info(f"Question: {example['question']}")
        logger.info(f"Aggregate Score: {example['aggregate_score']:.4f}")

if __name__ == "__main__":
    main()
