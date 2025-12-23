"""
Episode runner for the RL-style evaluation environment.

This module orchestrates the complete lifecycle:
1. Initialize workspace from template
2. Run agent on task (with mock mode for isolated environments)
3. Run automated grading checks
4. Run LLM judge evaluation
5. Save results and print summary
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
# This allows the script to be run directly: python env/runner.py
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import shutil
from datetime import datetime
from typing import Dict, Optional, Any
import json
import yaml

from env.sandbox import Sandbox
from agent.react_agent import ReActAgent
from grader.grade import Grader
from grader.rubric_judge import RubricJudge


class EpisodeRunner:
    """
    Manages a complete episode: workspace setup, agent execution, and grading.

    Workflow:
    1. Init: Create timestamped workspace from template
    2. Agent Loop: Run agent on task with mock mode
    3. Grading: Run automated checks (install, build, server health)
    4. Judge: Run LLM-based code evaluation
    """

    def __init__(
        self,
        template_name: str = "nextjs-starter",
        model_name: str = "gpt-4o-mini",
        max_steps: int = 50,
        verbose: bool = True
    ):
        """
        Initialize the episode runner.

        Args:
            template_name: Name of template in templates/ directory
            model_name: Model identifier from configs/models.yaml
            max_steps: Maximum agent steps
            verbose: Whether to print verbose logs
        """
        self.template_name = template_name
        self.model_name = model_name
        self.max_steps = max_steps
        self.verbose = verbose

        # Paths (relative to project root)
        self.project_root = Path(__file__).parent.parent
        self.template_path = self.project_root / "templates" / template_name
        self.runs_dir = self.project_root / "runs"

        # Episode-specific paths (set during init_workspace)
        self.episode_dir: Optional[Path] = None
        self.workspace_dir: Optional[Path] = None
        self.logs_dir: Optional[Path] = None
        self.agent_log_path: Optional[Path] = None
        self.system_log_path: Optional[Path] = None
        self.grader_log_path: Optional[Path] = None

        # Components (initialized during run)
        self.sandbox: Optional[Sandbox] = None
        self.agent: Optional[ReActAgent] = None

    def _log(self, message: str, prefix: str = "→") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"{prefix} {message}")

    def _construct_mock_mode_prompt(
        self,
        prompt: str,
        addition_for_cli_tools: str = ""
    ) -> str:
        """
        Construct the final prompt with mock mode instructions.

        Combines:
        1. Product Requirements (from dataset 'Prompt' column)
        2. Technical Constraints (from 'Addition for CLI Tools' column)
        3. System Override (mock mode for isolated environment)

        Args:
            prompt: Main product requirements
            addition_for_cli_tools: Technical constraints (e.g., "Use Supabase")

        Returns:
            Complete prompt with mock mode instructions
        """
        mock_mode_override = """

--- SYSTEM OVERRIDE: ISOLATED ENVIRONMENT MODE ---
You are running in a restricted demonstration environment with NO access to external APIs, Cloud Databases (Supabase), or Payment Gateways.

1. IGNORE requirements to use Supabase, Firebase, or external databases.
   - Instead, implement a **Local Mock Database**. Use a local JSON file (e.g., `data.json`) or browser `localStorage` to persist data.
   - For Authentication: Create a simple mock auth system (e.g., accept any email/password, store user in Context/Session).

2. IGNORE requirements for External APIs (Stock Data, Search, OpenAI, Stripe).
   - Instead, implement **Mock Services**.
   - Example: For stock prices, write a function that generates random fluctuating numbers.
   - Example: For payments, create a form that always returns "Success" after a 2-second delay.

Your goal is to build a FUNCTIONAL UI that demonstrates the logic, using simulated data and services."""

        # Combine all parts
        full_prompt = prompt

        if addition_for_cli_tools and addition_for_cli_tools.strip():
            full_prompt += f"\n\n--- TECHNICAL CONSTRAINTS ---\n{addition_for_cli_tools}"

        full_prompt += mock_mode_override

        return full_prompt

    def init_workspace(self) -> Path:
        """
        Initialize a fresh workspace from the template.

        Creates:
        - runs/<timestamp>/
        - runs/<timestamp>/workspace/ (copied from template)
        - runs/<timestamp>/logs/

        Returns:
            Path to the workspace directory

        Raises:
            FileNotFoundError: If template directory doesn't exist
        """
        # Validate template exists
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {self.template_path}\n\n"
                f"Please create a Next.js starter template:\n"
                f"  1. cd templates/\n"
                f"  2. pnpm create next-app nextjs-starter\n"
                f"  3. Follow the prompts to configure your template\n\n"
                f"Or create the directory manually if you have a custom template."
            )

        if not self.template_path.is_dir():
            raise ValueError(f"Template path is not a directory: {self.template_path}")

        # Create timestamped episode directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = self.runs_dir / timestamp
        self.workspace_dir = self.episode_dir / "workspace"
        self.logs_dir = self.episode_dir / "logs"

        # Create runs directory if it doesn't exist
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Create episode directory structure
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create log file paths
        self.agent_log_path = self.logs_dir / "agent.log"
        self.system_log_path = self.logs_dir / "system.log"
        self.grader_log_path = self.logs_dir / "grader.log"

        self._log(f"Initializing workspace from template: {self.template_name}", prefix="📁")
        self._log(f"Episode directory: {self.episode_dir}", prefix="  ")

        # Copy template to workspace
        try:
            shutil.copytree(
                self.template_path,
                self.workspace_dir,
                symlinks=False,
                ignore=shutil.ignore_patterns("node_modules", ".next", ".git"),
                dirs_exist_ok=False
            )
            self._log(f"✓ Template copied to: {self.workspace_dir}", prefix="  ")
        except Exception as e:
            # Clean up if copy fails
            if self.episode_dir.exists():
                shutil.rmtree(self.episode_dir)
            raise Exception(f"Failed to copy template: {e}")

        # Log workspace contents
        file_count = sum(1 for _ in self.workspace_dir.rglob("*") if _.is_file())
        self._log(f"✓ Workspace initialized with {file_count} files", prefix="  ")

        return self.workspace_dir

    def run_episode(
        self,
        task: str,
        rubric: Optional[str] = None,
        app_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a complete episode: workspace init + agent execution + grading.

        Args:
            task: Task description/prompt for the agent (already includes mock mode)
            rubric: Grading rubric (if None, uses default)
            app_name: Application name (for metadata)

        Returns:
            Dictionary with episode results
        """
        self._log("="*60, prefix="")
        self._log("STARTING EPISODE", prefix="🚀")
        if app_name:
            self._log(f"App: {app_name}", prefix="  ")
        self._log("="*60, prefix="")

        # Step 1: Initialize workspace
        self._log("\n[1/4] Initializing Workspace", prefix="📋")
        workspace = self.init_workspace()

        # Step 2: Run agent
        self._log("\n[2/4] Running Agent", prefix="🤖")
        self._log(f"Model: {self.model_name}", prefix="  ")
        self._log(f"Max steps: {self.max_steps}", prefix="  ")
        self._log(f"Task: {task[:100]}...", prefix="  ")

        # Initialize sandbox and agent
        self.sandbox = Sandbox(workspace)
        self.agent = ReActAgent(
            sandbox=self.sandbox,
            model_name=self.model_name,
            max_steps=self.max_steps,
            verbose=self.verbose,
            agent_log_path=self.agent_log_path,
            system_log_path=self.system_log_path
        )

        try:
            # Run agent
            agent_result = self.agent.run(task)

            # Step 3: Run Grading
            self._log("\n[3/4] Running Automated Checks", prefix="📊")
            grader = Grader(
                str(self.workspace_dir),
                grader_log_path=self.grader_log_path
            )
            grader_results = grader.run_all_checks()

            # Step 4: Run LLM Judge
            self._log("\n[4/4] Running LLM Judge", prefix="⚖️")
            judge = RubricJudge(model=self.model_name)

            # Use provided rubric or default
            if rubric is None:
                rubric = self._get_default_rubric()

            judge_results = judge.evaluate(
                workspace_path=str(self.workspace_dir),
                prompt=task,
                rubric=rubric
            )

            # Combine all results
            grade_result = {
                "automated_checks": grader_results,
                "llm_evaluation": judge_results,
                "overall_score": judge_results.get("score", 0),
                "overall_pass": grader_results.get("overall_pass", False) and judge_results.get("score", 0) >= 60
            }

            # Save grade result
            self._save_grade_result(grade_result)

            # Generate human-readable report
            self._generate_report_md(agent_result, grade_result, app_name, task)

            # Prepare episode result
            episode_result = {
                "episode_dir": str(self.episode_dir),
                "workspace_dir": str(self.workspace_dir),
                "logs_dir": str(self.logs_dir),
                "template": self.template_name,
                "model": self.model_name,
                "app_name": app_name,
                "task": task,
                "agent_result": agent_result,
                "grade_result": grade_result,
                "timestamp": self.episode_dir.name
            }

            # Save episode result
            self._save_episode_result(episode_result)

            # Print final summary
            self._print_final_summary(agent_result, grade_result, app_name)

            return episode_result

        finally:
            # CRITICAL: Clean up background processes (e.g., dev servers)
            # This prevents zombie Node.js processes from running indefinitely
            if self.sandbox:
                self._log("\nCleaning up background processes...", prefix="🧹")
                self.sandbox.cleanup()
                self._log("✓ Cleanup complete", prefix="  ")

    def _get_default_rubric(self) -> str:
        """
        Get the default grading rubric for evaluating agent work.

        Returns:
            Rubric string with evaluation criteria
        """
        return """Evaluate the Next.js application on the following criteria:

1. Functionality (40 points)
   - All required features are implemented
   - Features work correctly without errors
   - State management is correct
   - User interactions behave as expected

2. Code Quality (30 points)
   - Clean, readable, and well-organized code
   - Proper TypeScript types (no 'any' unless necessary)
   - Good component structure and separation of concerns
   - No unnecessary complexity or code duplication
   - Follows React and Next.js best practices

3. UI/UX (20 points)
   - Clean, modern, and professional design
   - Good use of Tailwind CSS
   - Responsive layout works on different screen sizes
   - Accessible (proper semantic HTML, ARIA labels where needed)
   - Good visual hierarchy and spacing

4. Production Readiness (10 points)
   - No console errors or warnings
   - Code is production-ready
   - Handles edge cases gracefully
   - Performance considerations (e.g., no unnecessary re-renders)

Provide a breakdown of scores for each criterion and an overall score (0-100)."""

    def _save_episode_result(self, result: Dict[str, Any]) -> None:
        """Save episode result to JSON file."""
        result_file = self.episode_dir / "result.json"

        # Make result JSON-serializable
        serializable_result = {
            k: v for k, v in result.items()
            if k != "agent"  # Don't serialize agent object
        }

        with open(result_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)

        self._log(f"✓ Result saved to: {result_file}", prefix="  ")

    def _save_grade_result(self, grade_result: Dict[str, Any]) -> None:
        """Save grading result to grade.json file."""
        grade_file = self.episode_dir / "grade.json"

        with open(grade_file, 'w') as f:
            json.dump(grade_result, f, indent=2)

        self._log(f"✓ Grade saved to: {grade_file}", prefix="  ")

    def _generate_report_md(
        self,
        agent_result: Dict[str, Any],
        grade_result: Dict[str, Any],
        app_name: Optional[str],
        task: str
    ) -> None:
        """Generate a human-readable markdown report."""
        report_file = self.episode_dir / "report.md"

        # Build report content
        report_lines = [
            f"# Episode Report: {app_name or 'Application'}",
            "",
            f"**Timestamp:** {self.episode_dir.name}",
            f"**Model:** {self.model_name}",
            f"**Template:** {self.template_name}",
            "",
            "---",
            "",
            "## Task Description",
            "",
            task,
            "",
            "---",
            "",
            "## Agent Performance",
            "",
            f"- **Status:** {'✅ Success' if agent_result.get('success') else '❌ Failed'}",
            f"- **Steps Taken:** {agent_result.get('steps', 0)}/{self.max_steps}",
            f"- **Model:** {self.model_name}",
            "",
        ]

        # Add agent action summary if available
        if agent_result.get("actions"):
            report_lines.extend([
                "### Agent Actions",
                "",
                "| Step | Tool | Description |",
                "|------|------|-------------|"
            ])
            for idx, action in enumerate(agent_result.get("actions", [])[:20], 1):  # Limit to first 20
                tool_name = action.get("tool", "unknown")
                # Extract brief description from tool args
                args = action.get("args", {})
                if tool_name == "write_file":
                    desc = f"Write {args.get('path', 'file')}"
                elif tool_name == "read_file":
                    desc = f"Read {args.get('path', 'file')}"
                elif tool_name == "run_command":
                    desc = f"Run: {args.get('command', 'cmd')[:50]}"
                elif tool_name == "install_deps":
                    desc = "Install dependencies"
                elif tool_name == "start_server":
                    desc = "Start dev server"
                elif tool_name == "finish_task":
                    desc = "Complete task"
                else:
                    desc = str(args)[:50]
                report_lines.append(f"| {idx} | `{tool_name}` | {desc} |")
            report_lines.extend(["", ""])

        # Automated checks section
        checks = grade_result.get("automated_checks", {})
        report_lines.extend([
            "## Automated Checks",
            "",
            f"- **Install:** {'✅ Pass' if checks.get('install') else '❌ Fail'}",
            f"- **Build:** {'✅ Pass' if checks.get('build') else '❌ Fail'}",
            f"- **Server Health:** {'✅ Pass' if checks.get('server_health') else '❌ Fail'}",
            f"- **Overall:** {'✅ Pass' if checks.get('overall_pass') else '❌ Fail'}",
            "",
        ])

        # LLM evaluation section
        llm_eval = grade_result.get("llm_evaluation", {})
        report_lines.extend([
            "## LLM Judge Evaluation",
            "",
            f"**Overall Score:** {llm_eval.get('score', 0)}/100",
            "",
        ])

        # Add breakdown if available
        breakdown = llm_eval.get("breakdown", {})
        if breakdown:
            report_lines.extend([
                "### Criteria Breakdown",
                ""
            ])
            for criterion, score in breakdown.items():
                report_lines.append(f"- **{criterion}:** {score}/100")
            report_lines.append("")

        # Add reasoning if available
        if llm_eval.get("reasoning"):
            report_lines.extend([
                "### Judge Reasoning",
                "",
                llm_eval.get("reasoning"),
                "",
            ])

        # Overall result
        overall_pass = grade_result.get("overall_pass", False)
        overall_score = grade_result.get("overall_score", 0)
        report_lines.extend([
            "---",
            "",
            "## Final Result",
            "",
            f"- **Status:** {'✅ PASS' if overall_pass else '❌ FAIL'}",
            f"- **Final Score:** {overall_score}/100",
            "",
        ])

        # Links to files
        report_lines.extend([
            "---",
            "",
            "## Files",
            "",
            f"- **Workspace:** `{self.workspace_dir.relative_to(self.project_root)}`",
            f"- **Logs:** `{self.logs_dir.relative_to(self.project_root)}`",
            f"  - Agent Log: `logs/agent.log`",
            f"  - System Log: `logs/system.log`",
            f"  - Grader Log: `logs/grader.log`",
            f"- **Results:**",
            f"  - Episode Result: `result.json`",
            f"  - Grade Result: `grade.json`",
            "",
        ])

        # Write report
        report_content = "\n".join(report_lines)
        with open(report_file, 'w') as f:
            f.write(report_content)

        self._log(f"✓ Report saved to: {report_file}", prefix="  ")

    def _print_final_summary(
        self,
        agent_result: Dict[str, Any],
        grade_result: Dict[str, Any],
        app_name: Optional[str] = None
    ) -> None:
        """Print a nice summary table of the episode results."""
        print("\n" + "="*70)
        print(" "*25 + "EPISODE SUMMARY")
        print("="*70)

        if app_name:
            print(f"\n📱 APP: {app_name}")
            print("-" * 70)

        # Agent section
        print("\n📋 AGENT PERFORMANCE")
        print("-" * 70)
        agent_status = "✅ SUCCESS" if agent_result.get("success") else "❌ FAILED"
        print(f"  Status:       {agent_status}")
        print(f"  Steps:        {agent_result.get('steps', 0)}/{self.max_steps}")
        print(f"  Model:        {self.model_name}")

        # Automated checks section
        print("\n🔧 AUTOMATED CHECKS")
        print("-" * 70)
        checks = grade_result.get("automated_checks", {})
        install_status = "✅ PASS" if checks.get("install") else "❌ FAIL"
        build_status = "✅ PASS" if checks.get("build") else "❌ FAIL"
        server_status = "✅ PASS" if checks.get("server_health") else "❌ FAIL"
        print(f"  Install:      {install_status}")
        print(f"  Build:        {build_status}")
        print(f"  Server:       {server_status}")

        # LLM evaluation section
        print("\n⚖️  LLM EVALUATION")
        print("-" * 70)
        llm_eval = grade_result.get("llm_evaluation", {})
        score = llm_eval.get("score", 0)
        print(f"  Score:        {score}/100")

        # Show breakdown if available
        breakdown = llm_eval.get("breakdown", {})
        if breakdown:
            print(f"  Breakdown:")
            for criterion, criterion_score in breakdown.items():
                print(f"    - {criterion}: {criterion_score}")

        # Overall result
        print("\n🏆 OVERALL RESULT")
        print("-" * 70)
        overall_pass = grade_result.get("overall_pass", False)
        overall_status = "✅ PASS" if overall_pass else "❌ FAIL"
        print(f"  Status:       {overall_status}")
        print(f"  Final Score:  {score}/100")

        # Cost (if available in agent_result)
        if "total_cost" in agent_result:
            print(f"  Total Cost:   ${agent_result['total_cost']:.4f}")

        print("\n📂 OUTPUT")
        print("-" * 70)
        print(f"  Episode:      {self.episode_dir}")
        print(f"  Workspace:    {self.workspace_dir}")
        print(f"  Results:      {self.episode_dir}/result.json")
        print(f"  Grade:        {self.episode_dir}/grade.json")

        print("\n" + "="*70 + "\n")

    def get_workspace_path(self) -> Optional[Path]:
        """Get the current workspace path."""
        return self.workspace_dir

    def get_episode_path(self) -> Optional[Path]:
        """Get the current episode directory path."""
        return self.episode_dir

    def cleanup(self, keep_workspace: bool = True) -> None:
        """
        Clean up episode resources.

        Args:
            keep_workspace: If False, delete the entire episode directory
        """
        if not keep_workspace and self.episode_dir and self.episode_dir.exists():
            self._log(f"Cleaning up episode: {self.episode_dir}", prefix="🗑️")
            shutil.rmtree(self.episode_dir)
            self._log("✓ Episode directory deleted", prefix="  ")


def load_models_config() -> Dict[str, Any]:
    """
    Load model configurations from configs/models.yaml.

    Returns:
        Dictionary with model configurations
    """
    config_path = Path(__file__).parent.parent / "configs" / "models.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Models config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_task_from_csv(csv_path: str, row_index: int = 0) -> Dict[str, str]:
    """
    Load a task from the HuggingFace dataset CSV.

    Expected CSV format with columns:
    - App Name: Name of the application
    - App Description: Brief description
    - Prompt: Main product requirements
    - Addition for CLI Tools: Technical constraints (e.g., "Use Supabase")
    - Rubric: Grading criteria

    Args:
        csv_path: Path to the CSV file
        row_index: Index of the row to load (default: 0)

    Returns:
        Dictionary with task data:
        {
            "app_name": str,
            "app_description": str,
            "prompt": str,
            "addition_for_cli_tools": str,
            "rubric": str
        }

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns not found or row index invalid
    """
    import csv

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        if not rows:
            raise ValueError(f"CSV file is empty: {csv_path}")

        # Check for required columns
        required_cols = ['App Name', 'Prompt', 'Rubric']
        missing_cols = [col for col in required_cols if col not in rows[0]]
        if missing_cols:
            available_cols = ', '.join(rows[0].keys())
            raise ValueError(
                f"CSV missing required columns: {missing_cols}\n"
                f"Available columns: {available_cols}"
            )

        if row_index < 0 or row_index >= len(rows):
            raise ValueError(
                f"Row index {row_index} out of range. CSV has {len(rows)} rows (0-{len(rows)-1})"
            )

        row = rows[row_index]

        return {
            "app_name": row.get('App Name', ''),
            "app_description": row.get('App Description', ''),
            "prompt": row.get('Prompt', ''),
            "addition_for_cli_tools": row.get('Addition for CLI Tools', ''),
            "rubric": row.get('Rubric', '')
        }


def run_episode(
    task: str,
    template_name: str = "nextjs-starter",
    model_name: str = "gpt-4o-mini",
    max_steps: int = 50,
    verbose: bool = True,
    rubric: Optional[str] = None,
    app_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run a single episode.

    Args:
        task: Task description/prompt (with mock mode instructions)
        template_name: Template to use
        model_name: Model identifier
        max_steps: Maximum agent steps
        verbose: Whether to print logs
        rubric: Grading rubric (optional)
        app_name: Application name (optional)

    Returns:
        Episode result dictionary
    """
    runner = EpisodeRunner(
        template_name=template_name,
        model_name=model_name,
        max_steps=max_steps,
        verbose=verbose
    )
    return runner.run_episode(task=task, rubric=rubric, app_name=app_name)


if __name__ == "__main__":
    import argparse
    from configs.load_env import load_env

    load_env()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an RL-style evaluation episode for LLM agent building web apps"
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="Task description/prompt (or use --data to load from CSV)"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to CSV file with HuggingFace dataset format"
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Row index to load from CSV (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier from configs/models.yaml (e.g., gemini-flash, claude-sonnet, gpt-4o-mini)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="nextjs-starter",
        help="Template name from templates/ directory (default: nextjs-starter)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum agent steps (default: 50)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logging"
    )

    args = parser.parse_args()

    # Load models config
    try:
        models_config = load_models_config()
        default_model_key = models_config.get('default', 'gemini-flash')
        print(f"✓ Loaded models config (default: {default_model_key})")
    except Exception as e:
        print(f"Error: Could not load models config: {e}")
        sys.exit(1)

    # Resolve model name
    # Use specified model or default from config
    model_key = args.model if args.model else default_model_key

    # Look up the litellm model identifier
    if model_key not in models_config.get('models', {}):
        print(f"Error: Model '{model_key}' not found in configs/models.yaml")
        print(f"Available models: {', '.join(models_config.get('models', {}).keys())}")
        sys.exit(1)

    # Get the litellm model identifier (e.g., "gemini/gemini-2.0-flash-001")
    litellm_model = models_config['models'][model_key]['litellm_params']['model']
    print(f"✓ Using model: {model_key} → {litellm_model}")

    # Determine task source
    if args.data:
        # Load from HuggingFace dataset CSV
        try:
            task_data = load_task_from_csv(args.data, args.row_index)
            print(f"✓ Loaded task from {args.data} (row {args.row_index})")
            print(f"  App: {task_data['app_name']}")

            # Create runner instance to access mock mode prompt builder
            runner = EpisodeRunner(
                template_name=args.template,
                model_name=litellm_model,
                max_steps=args.max_steps,
                verbose=not args.quiet
            )

            # Construct mock mode prompt
            full_task = runner._construct_mock_mode_prompt(
                prompt=task_data['prompt'],
                addition_for_cli_tools=task_data['addition_for_cli_tools']
            )

            # Run episode
            result = runner.run_episode(
                task=full_task,
                rubric=task_data['rubric'],
                app_name=task_data['app_name']
            )

        except Exception as e:
            print(f"Error loading from CSV: {e}")
            sys.exit(1)
    elif args.task:
        # Use command-line task
        task = " ".join(args.task)
        result = run_episode(
            task=task,
            template_name=args.template,
            model_name=litellm_model,
            max_steps=args.max_steps,
            verbose=not args.quiet
        )
    else:
        # Default task
        task = "Create a simple homepage with a welcome message."
        result = run_episode(
            task=task,
            template_name=args.template,
            model_name=litellm_model,
            max_steps=args.max_steps,
            verbose=not args.quiet
        )

    print("\n✅ Episode completed successfully!")
    print(f"   Results saved to: {result['episode_dir']}")
