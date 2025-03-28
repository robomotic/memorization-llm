# Running Memorization Detection Experiments

This guide provides step-by-step instructions for setting up and running the memorization detection experiments on the MedQA-USMLE dataset using the implemented methods.

## Prerequisites

- Python 3.8 or higher
- Internet connection for downloading datasets and model weights
- OpenRouter API key (for accessing LLM APIs)

## Setup Instructions

### 1. Environment Setup

First, set up the virtual environment and install the required dependencies:

```bash
# Navigate to the project directory
cd /path/to/MemorizationLLM

# Create and activate the virtual environment
python -m venv memorization_env
source memorization_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

You can use either OpenRouter or Azure OpenAI as your API backend. Configure the appropriate settings in the `.env` file:

```bash
# Edit the .env file
nano .env
```

#### OpenRouter Configuration

To use OpenRouter (default):

```bash
# Update the API key (replace with your actual key)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Make sure the backend is set to openrouter
API_BACKEND=openrouter
```

You can obtain an OpenRouter API key by signing up at [openrouter.ai](https://openrouter.ai).

#### Azure OpenAI Configuration

To use Azure OpenAI:

```bash
# Update Azure OpenAI settings
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# Set the backend to azure
API_BACKEND=azure
```

You'll need an Azure OpenAI service resource with a deployed model. The deployment name should be specified in the `AZURE_OPENAI_DEPLOYMENT` variable.

### 3. Dataset Preparation

The code will automatically download and cache the MedQA-USMLE dataset from Hugging Face. The first run might take some time as it downloads the dataset.

## Running Experiments

### Basic Usage

The simplest way to run the experiments is:

```bash
# Activate the environment (if not already activated)
source memorization_env/bin/activate

# Run with default settings (10 examples from train split)
python main.py
```

This will:
1. Load 10 random examples from the MedQA-USMLE training set
2. Run all four detection methods
3. Save results to the `results/` directory

### Customizing Experiments

You can customize the experiments using various command-line arguments:

```bash
python main.py [OPTIONS]
```

Available options:

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model identifier to use | `openai/gpt-4-turbo` |
| `--api-backend` | API backend to use (openrouter or azure) | From `.env` file |
| `--split` | Dataset split (train, validation, test) | `train` |
| `--num_examples` | Number of examples to analyze | `10` |
| `--methods` | Detection methods to run | All methods |
| `--output` | Custom output file path | Auto-generated |
| `--seed` | Random seed for reproducibility | `42` |

### Example Commands

#### Run with a specific model:
```bash
python main.py --model anthropic/claude-3-sonnet-20240229
```

#### Run on validation split with more examples:
```bash
python main.py --split validation --num_examples 20
```

#### Run only specific detection methods:
```bash
python main.py --methods perplexity embedding
```

#### Run a quick test with fewer examples:
```bash
python main.py --num_examples 3 --methods perplexity
```

## Understanding Results

Results are saved as JSON files in the `results/` directory. Each file contains:

1. **Metadata**: Information about the experiment run
2. **Results**: For each example:
   - Question and options
   - Results from each detection method
   - Aggregate memorization score
   - Overall memorization verdict

Example structure:
```json
{
  "metadata": {
    "model": "openai/gpt-4-turbo",
    "split": "train",
    "num_examples": 10,
    "methods": ["perplexity", "ngram", "embedding", "consistency"],
    "timestamp": "2025-03-27T11:00:00.000Z"
  },
  "results": [
    {
      "id": "example_id",
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_answer": "A",
      "detection_results": {
        "perplexity": { ... },
        "ngram": { ... },
        "embedding": { ... },
        "consistency": { ... }
      },
      "aggregate_score": 0.75,
      "is_memorized": true
    },
    ...
  ]
}
```

## Analyzing Results

You can analyze the results using the following approaches:

### 1. Summary Statistics

The script outputs basic summary statistics after completion:
- Number of examples detected as memorized
- Examples of memorized and non-memorized questions

### 2. Custom Analysis

For more detailed analysis, you can write custom scripts to process the JSON results:

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('results/medqa_memorization_openai_gpt-4-turbo_20250327_110000.json', 'r') as f:
    data = json.load(f)

# Extract scores
scores = [r['aggregate_score'] for r in data['results']]

# Plot distribution
plt.hist(scores, bins=10)
plt.xlabel('Memorization Score')
plt.ylabel('Count')
plt.title('Distribution of Memorization Scores')
plt.savefig('memorization_distribution.png')
```

## Troubleshooting

### API Rate Limits

If you encounter rate limit errors from OpenRouter:
- Reduce the number of examples (`--num_examples`)
- Use fewer detection methods (`--methods`)
- Add delays between API calls (modify `src/utils/api.py`)

### Memory Issues

If you encounter memory issues with the embedding model:
- Use a smaller embedding model (modify `EMBEDDING_MODEL` in `src/config.py`)
- Process fewer examples at a time

### Dataset Access Issues

If you have trouble accessing the MedQA dataset:
- Check your internet connection
- Verify Hugging Face is accessible
- Try downloading the dataset manually and placing it in the `data/cache` directory

## Advanced Configuration

For advanced users, you can modify the configuration parameters in `src/config.py`:

- `PERPLEXITY_THRESHOLD`: Threshold for perplexity-based detection
- `NGRAM_SIZES`: N-gram sizes to analyze
- `EMBEDDING_MODEL`: Model for embedding-based detection
- `EMBEDDING_SIMILARITY_THRESHOLD`: Threshold for embedding similarity
- `CONSISTENCY_TRIALS`: Number of trials for consistency testing

## Citation

If you use this code in your research, please cite the relevant papers listed in the main README.md file.
