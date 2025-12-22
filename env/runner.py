"""
Episode runner for the RL-style evaluation environment.

This module orchestrates the complete lifecycle:
1. Initialize workspace from template
2. Run agent on task
3. Collect results and logs
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

from env.sandbox import Sandbox
from agent.react_agent import ReActAgent


class EpisodeRunner:
    """
    Manages a complete episode: workspace setup, agent execution, and cleanup.

    Workflow:
    1. Init: Create timestamped workspace from template
    2. Agent Loop: Run agent on task
    3. Grading: (Future) Run automated checks and rubric judge
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
            model_name: LiteLLM model identifier
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

        # Components (initialized during run)
        self.sandbox: Optional[Sandbox] = None
        self.agent: Optional[ReActAgent] = None

    def _log(self, message: str, prefix: str = "→") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"{prefix} {message}")

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

    def run_episode(self, task: str) -> Dict[str, Any]:
        """
        Run a complete episode: workspace init + agent execution.

        Args:
            task: Task description/prompt for the agent

        Returns:
            Dictionary with episode results
        """
        self._log("="*60, prefix="")
        self._log("STARTING EPISODE", prefix="🚀")
        self._log("="*60, prefix="")

        # Step 1: Initialize workspace
        self._log("\n[1/2] Initializing Workspace", prefix="📋")
        workspace = self.init_workspace()

        # Step 2: Run agent
        self._log("\n[2/2] Running Agent", prefix="🤖")
        self._log(f"Model: {self.model_name}", prefix="  ")
        self._log(f"Max steps: {self.max_steps}", prefix="  ")
        self._log(f"Task: {task}", prefix="  ")

        # Initialize sandbox and agent
        self.sandbox = Sandbox(workspace)
        self.agent = ReActAgent(
            sandbox=self.sandbox,
            model_name=self.model_name,
            max_steps=self.max_steps,
            verbose=self.verbose
        )

        try:
            # Run agent
            agent_result = self.agent.run(task)

            # Prepare episode result
            episode_result = {
                "episode_dir": str(self.episode_dir),
                "workspace_dir": str(self.workspace_dir),
                "logs_dir": str(self.logs_dir),
                "template": self.template_name,
                "model": self.model_name,
                "task": task,
                "agent_result": agent_result,
                "timestamp": self.episode_dir.name
            }

            # Save episode result
            self._save_episode_result(episode_result)

            self._log("\n" + "="*60, prefix="")
            self._log("EPISODE COMPLETE", prefix="✅")
            self._log("="*60, prefix="")
            self._log(f"Results saved to: {self.episode_dir}", prefix="  ")

            return episode_result

        finally:
            # CRITICAL: Clean up background processes (e.g., dev servers)
            # This prevents zombie Node.js processes from running indefinitely
            if self.sandbox:
                self._log("\nCleaning up background processes...", prefix="🧹")
                self.sandbox.cleanup()
                self._log("✓ Cleanup complete", prefix="  ")

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


def run_episode(
    task: str,
    template_name: str = "nextjs-starter",
    model_name: str = "gpt-4o-mini",
    max_steps: int = 50,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run a single episode.

    Args:
        task: Task description/prompt
        template_name: Template to use
        model_name: LiteLLM model identifier
        max_steps: Maximum agent steps
        verbose: Whether to print logs

    Returns:
        Episode result dictionary
    """
    runner = EpisodeRunner(
        template_name=template_name,
        model_name=model_name,
        max_steps=max_steps,
        verbose=verbose
    )
    return runner.run_episode(task)


def load_prompt_from_csv(csv_path: str, row_index: int = 0) -> str:
    """
    Load a prompt from a CSV file.

    Expected CSV format (HuggingFace dataset format):
    - Should have a 'prompt' column containing the task description

    Args:
        csv_path: Path to the CSV file
        row_index: Index of the row to load (default: 0)

    Returns:
        Prompt string from the specified row

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If prompt column not found or row index invalid
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

        if 'prompt' not in rows[0]:
            available_cols = ', '.join(rows[0].keys())
            raise ValueError(
                f"CSV must have a 'prompt' column. Available columns: {available_cols}"
            )

        if row_index < 0 or row_index >= len(rows):
            raise ValueError(
                f"Row index {row_index} out of range. CSV has {len(rows)} rows (0-{len(rows)-1})"
            )

        return rows[row_index]['prompt']


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

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
        help="Path to CSV file with prompts (expects 'prompt' column)"
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
        default="gpt-4o-mini",
        help="LiteLLM model identifier (default: gpt-4o-mini)"
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

    # Determine task source
    if args.data:
        # Load from CSV
        try:
            task = load_prompt_from_csv(args.data, args.row_index)
            print(f"Loaded prompt from {args.data} (row {args.row_index})")
        except Exception as e:
            print(f"Error loading from CSV: {e}")
            sys.exit(1)
    elif args.task:
        # Use command-line task
        task = " ".join(args.task)
    else:
        # Default task
        task = "Create a simple homepage with a welcome message."

    # Run episode
    result = run_episode(
        task=task,
        template_name=args.template,
        model_name=args.model,
        max_steps=args.max_steps,
        verbose=not args.quiet
    )

    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    print(f"Success: {result['agent_result']['success']}")
    print(f"Steps: {result['agent_result']['steps']}")
    print(f"Workspace: {result['workspace_dir']}")
