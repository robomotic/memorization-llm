"""
Configuration settings for the memorization detection system.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Backend Selection
API_BACKEND = os.getenv("API_BACKEND", "openrouter").lower()
if API_BACKEND not in ["openrouter", "azure"]:
    raise ValueError(f"Invalid API_BACKEND: {API_BACKEND}. Must be 'openrouter' or 'azure'.")

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Default model to use
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4-turbo")

# Dataset Configuration
MEDQA_DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")

# Detection Method Parameters
PERPLEXITY_THRESHOLD = 1.5  # Ratio threshold for perplexity-based detection
NGRAM_SIZES = [2, 3, 4]  # N-gram sizes to analyze
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Model for embedding-based detection
EMBEDDING_SIMILARITY_THRESHOLD = 0.85  # Threshold for embedding similarity detection
CONSISTENCY_TRIALS = 5  # Number of trials for consistency testing

# Output Configuration
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
