# AfterQuery RL-Style Evaluation Environment

A reproducible RL-style evaluation harness for measuring how well LLMs act as autonomous agents to create working "0→1" web applications from product prompts—covering planning, coding, debugging, running, and shipping.

Slides: https://docs.google.com/presentation/d/1qjJMSwOZm5m8hZjeSUO6GwXu9gDM9bZWFiFpASdD5sg/edit?usp=sharing

Demo: <insert-video>

## Quick Start

### Prerequisites

- **Python 3.9+**
- **Node.js 18+** and **pnpm**
- **API Keys**: OpenAI, Claude, or Gemini

### Installation

1. **Clone the repository:**
```bash
git clone <repo-url>
cd RL-environment
```

2. **Install Python dependencies:**
```bash
pip install -e .
```

This installs the project in editable mode from `pyproject.toml`.

3. **Set up environment variables:**
Copy the example config and add your API keys:
```bash
cp configs/env.yaml.example configs/env.yaml
# Then edit configs/env.yaml and add your actual API keys
```

Example `configs/env.yaml`:
```yaml
openai_api_key: "sk-..."
anthropic_api_key: "sk-ant-..."
google_api_key: "..."
```

4. **Install Node.js tools:**
```bash
npm install -g pnpm
```

### Run Your First Episode

**Simple test:**
```bash
python3 env/runner.py --data data/prompts_updated.csv --row-index 0
```

**With specific model:**
```bash
python3 env/runner.py --data data/prompts_updated.csv --model claude-sonnet
```

## Usage

### Command-Line Interface

```bash
python3 env/runner.py [OPTIONS] [TASK]
```

**Options:**
- `--data PATH`: Path to CSV file with HuggingFace dataset format
- `--row-index N`: Row index to load from CSV (default: 0)
- `--model NAME`: Model identifier from `configs/models.yaml` (default: gemini-flash)
- `--template NAME`: Template from `templates/` directory (default: nextjs-starter)
- `--max-steps N`: Maximum agent steps (default: 50)
- `--step-delay SECONDS`: Delay between consecutive API calls to avoid rate limits (default: 0.0)
- `--quiet`: Disable verbose logging

### Examples

**1. Run with original huggingface dataset
```bash
python3 env/runner.py --data data/prompts.csv --row-index 0
```

**2. Test different models:**
```bash
# Gemini 2.0 Flash (fast, default)
python3 env/runner.py --data data/prompts_updated.csv --row-index 1 --model gemini-flash

# Claude Sonnet 4.5 (best for coding)
python3 env/runner.py --data data/prompts_updated.csv --row-index 2 --model claude-sonnet

# GPT-4o-mini (fast and cheap)
python3 env/runner.py --data data/prompts_updated.csv --row-index 3 --model gpt-4o-mini
```

**3. Avoid rate limits with step delay:**
```bash
# Claude Sonnet with 4-second delay between API calls
python3 env/runner.py \
  --data data/prompts_updated.csv \
  --model claude-sonnet \
  --step-delay 4
```

**4. Run multiple episodes:**
```bash
# Evaluate models on first 5 tasks
for i in {0..4}; do
  python3 env/runner.py --data data/prompts.csv --row-index $i --model gemini-flash
done
```

## Dataset Format

The system expects a CSV file with the following columns (from [AfterQuery/App-Bench](https://huggingface.co/datasets/AfterQuery/App-Bench)):

| Column | Description |
|--------|-------------|
| `App Name` | Name of the application |
| `App Description` | Brief description |
| `Prompt` | Main product requirements |
| `Addition for CLI Tools` | Technical constraints (e.g., "Use Supabase") |
| `Rubric` | Grading criteria for LLM judge |

**Example:**
```csv
App Name,App Description,Prompt,Addition for CLI Tools,Rubric
Stock Tracker,Track stock portfolio,"Build a stock portfolio tracker...","Use localStorage for data","Evaluate on: 1) Functionality..."
```