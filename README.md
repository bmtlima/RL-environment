# AfterQuery RL-Style Evaluation Environment

A reproducible RL-style evaluation harness for measuring how well LLMs act as autonomous agents to create working "0→1" web applications from product prompts—covering planning, coding, debugging, running, and shipping.

## Overview

This environment evaluates LLM agents on their ability to:
- Plan implementation from requirements
- Write clean, functional code
- Debug and fix errors
- Build and deploy working applications
- Pass automated checks and quality reviews

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **pnpm**
- **API Keys**: OpenAI, Anthropic, or Google AI

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
Create a `.env` file:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

4. **Install Node.js tools:**
```bash
npm install -g pnpm
```

### Run Your First Episode

**Simple test:**
```bash
python3 env/runner.py "Build a counter app with increment and decrement buttons"
```

**With HuggingFace dataset:**
```bash
python3 env/runner.py --data data/prompts.csv --row-index 0
```

**With specific model:**
```bash
python3 env/runner.py --data data/prompts.csv --model claude-sonnet
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
- `--quiet`: Disable verbose logging

### Examples

**1. Run with HuggingFace dataset:**
```bash
python3 env/runner.py --data data/prompts.csv --row-index 0
```

**2. Test different models:**
```bash
# Gemini 2.0 Flash (fast, default)
python3 env/runner.py --data data/prompts.csv --model gemini-flash

# Claude Sonnet 4.5 (best for coding)
python3 env/runner.py --data data/prompts.csv --model claude-sonnet

# GPT-4o-mini (fast and cheap)
python3 env/runner.py --data data/prompts.csv --model gpt-4o-mini
```

**3. Custom task with specific settings:**
```bash
python3 env/runner.py \
  "Build a habit tracker with daily check-ins" \
  --model claude-sonnet \
  --max-steps 30
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

## Architecture

### Components

#### 1. **Agent Harness** (`env/`)
Manages the episode lifecycle and provides tools to the agent:

**Tools:**
- `write_file(path, content)` - Create/modify files
- `read_file(path)` - Read file contents
- `run_command(cmd, cwd)` - Execute shell commands
- `install_deps()` - Install Node.js dependencies with architecture detection
- `start_server()` - Start development server in background
- `finish_task()` - Signal task completion

**Files:**
- `runner.py` - Episode orchestration
- `tools.py` - Tool implementations
- `sandbox.py` - Process isolation and execution

#### 2. **Agent** (`agent/`)
LLM-powered autonomous coding agent using ReAct pattern:

**Strategy:**
- Thinks about the task
- Plans next action
- Executes tools
- Reflects on results
- Repeats until complete

**Files:**
- `react_agent.py` - ReAct agent implementation
- `prompts/system.txt` - System prompt with instructions
- `prompts/tool_schema.json` - Tool definitions for LLM

#### 3. **Grader** (`grader/`)
Hybrid evaluation with automated checks + LLM judge:

**Automated Checks:**
- ✅ Dependencies install successfully
- ✅ Application builds without errors
- ✅ Server starts and responds with HTTP 200

**LLM Judge:**
- 📊 Code quality and clarity
- 🎨 UI/UX polish
- 🎯 Product requirements fit
- 🛡️ Edge case handling

**Files:**
- `grade.py` - Automated checks (install, build, server health)
- `rubric_judge.py` - LLM-based code evaluation

### Episode Lifecycle

```
1. Init
   └─ Create fresh workspace from template

2. Agent Loop
   ├─ Read task prompt + workspace state
   ├─ Think and plan next action
   ├─ Execute tool calls
   ├─ Implement features
   ├─ Run tests and fix errors
   └─ Repeat until done or max steps

3. Grading
   ├─ Run install_deps()
   ├─ Run build (pnpm build)
   ├─ Check server health (pnpm start)
   ├─ Run LLM judge on code
   └─ Generate grade.json

4. Output
   ├─ runs/<timestamp>/workspace/  (agent's work)
   ├─ runs/<timestamp>/result.json (episode data)
   └─ runs/<timestamp>/grade.json  (grading results)
```

## Repository Structure

```
RL-environment/
├── README.md
├── pyproject.toml           # Python dependencies
├── .env                     # API keys (gitignored)
│
├── configs/
│   └── models.yaml          # Model configurations
│
├── env/                     # Agent Harness
│   ├── runner.py            # Episode orchestration
│   ├── tools.py             # Tool implementations
│   └── sandbox.py           # Process isolation
│
├── agent/                   # Autonomous Agent
│   ├── react_agent.py       # ReAct agent logic
│   └── prompts/
│       ├── system.txt       # System prompt
│       └── tool_schema.json # Tool definitions
│
├── grader/                  # Evaluation System
│   ├── grade.py             # Automated checks
│   └── rubric_judge.py      # LLM judge
│
├── templates/
│   └── nextjs-starter/      # Next.js 16 + Tailwind v3
│
├── data/
│   └── prompts.csv          # HuggingFace dataset
│
└── runs/                    # Episode outputs
    └── <timestamp>/
        ├── workspace/       # Agent's code
        ├── result.json      # Episode results
        └── grade.json       # Grading scores
```

## Configuration

### Adding New Models

Edit `configs/models.yaml`:

```yaml
models:
  your-model-name:
    model_name: "provider/model-id"
    litellm_params:
      model: "provider/model-id"
      temperature: 0.7
```

Supported providers via LiteLLM:
- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Anthropic: `anthropic/claude-sonnet-4-5-20250929`
- Google: `gemini/gemini-2.0-flash-001`
- OpenRouter: `openrouter/model-name`

**Current models in `configs/models.yaml`:**
- `gpt-4o-mini` - OpenAI GPT-4o-mini
- `claude-sonnet` - Anthropic Claude Sonnet 4.5
- `gemini-flash` - Google Gemini 2.0 Flash (default)

### Mock Mode (Isolated Environment)

The system runs agents in **Mock Mode** to avoid requiring external services:

**Automatically mocked:**
- 🗄️ Databases (Supabase, Firebase) → Use localStorage or JSON files
- 🔐 Authentication → Mock login system
- 💳 Payment APIs (Stripe) → Fake success responses
- 📊 External APIs (Stock data, Weather) → Random/mock data

This is automatically injected via the prompt, so agents build functional UIs without real API keys.

## Testing Components

### Test the Grader

```bash
# Test automated checks only
python3 test_grader.py runs/<timestamp>/workspace

# Test with server health check (slower)
python3 test_grader.py runs/<timestamp>/workspace --all
```

### Test the LLM Judge

```bash
python3 test_judge.py runs/<timestamp>/workspace
```

## Output & Results

After each episode, results are saved to `runs/<timestamp>/`:

### `result.json`
Complete episode data:
```json
{
  "episode_dir": "runs/20241223_143022",
  "model": "gemini-flash",
  "app_name": "Stock Portfolio Tracker",
  "task": "Build a stock tracker...",
  "agent_result": {
    "success": true,
    "steps": 12,
    "actions": [...]
  },
  "grade_result": {...}
}
```

### `grade.json`
Grading results:
```json
{
  "automated_checks": {
    "install": true,
    "build": true,
    "server_health": true,
    "overall_pass": true
  },
  "llm_evaluation": {
    "score": 85,
    "reasoning": "Clean implementation with good UI...",
    "breakdown": {
      "Functionality": 90,
      "Code Quality": 85,
      "UI/UX": 80,
      "Production Readiness": 85
    }
  },
  "overall_score": 85,
  "overall_pass": true
}
```

### Console Output

```
======================================================================
                         EPISODE SUMMARY
======================================================================

📱 APP: Stock Portfolio Tracker

📋 AGENT PERFORMANCE
----------------------------------------------------------------------
  Status:       ✅ SUCCESS
  Steps:        12/50
  Model:        gemini-flash

🔧 AUTOMATED CHECKS
----------------------------------------------------------------------
  Install:      ✅ PASS
  Build:        ✅ PASS
  Server:       ✅ PASS

⚖️  LLM EVALUATION
----------------------------------------------------------------------
  Score:        85/100
  Breakdown:
    - Functionality: 90
    - Code Quality: 85
    - UI/UX: 80
    - Production Readiness: 85

🏆 OVERALL RESULT
----------------------------------------------------------------------
  Status:       ✅ PASS
  Final Score:  85/100

📂 OUTPUT
----------------------------------------------------------------------
  Episode:      runs/20241223_143022
  Workspace:    runs/20241223_143022/workspace
  Results:      runs/20241223_143022/result.json
  Grade:        runs/20241223_143022/grade.json
```

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError"**
```bash
# Use python3 explicitly
python3 env/runner.py --data data/prompts.csv
```

**2. "Cannot find module lightningcss"**
- Fixed! We downgraded to Tailwind CSS v3 to avoid native binary issues

**3. "Server failed to start"**
- Check if port 3000 is already in use: `lsof -ti:3000 | xargs kill -9`
- Verify `pnpm` is installed: `npm install -g pnpm`

**4. API key not found**
```bash
# Ensure .env file exists with your keys
echo "OPENAI_API_KEY=sk-..." > .env
```

**5. Agent gets stuck in loop**
- Reduce `--max-steps` to force earlier termination
- Check `runs/<timestamp>/workspace` to see what agent built

## Development

### Project Structure

The codebase follows a clean architecture:

- **Harness** (`env/`) - Provides tools and manages episodes
- **Agent** (`agent/`) - LLM controller with ReAct loop
- **Grader** (`grader/`) - Automated + LLM evaluation
- **Configs** (`configs/`) - Model settings
- **Templates** (`templates/`) - Starting codebases

### Adding New Tools

1. Add tool to `env/tools.py`:
```python
def your_tool(self, arg: str) -> ToolResult:
    """Tool description."""
    # Implementation
    return ToolResult(success=True, data={...})
```

2. Update `agent/prompts/tool_schema.json`:
```json
{
  "name": "your_tool",
  "description": "What it does",
  "parameters": {...}
}
```