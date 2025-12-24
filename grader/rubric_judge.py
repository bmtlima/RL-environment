"""
Rubric-based LLM judge for evaluating agent-generated code.

This module provides subjective evaluation of code quality, UX polish,
product fit, and edge cases using an LLM as a judge.
"""

import os
import json
import re
import time
import math
from pathlib import Path
from typing import Dict, List
import litellm

# Load environment variables from configs/env.yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.load_env import load_env
load_env()


class RubricJudge:
    """
    LLM-based judge for subjective evaluation of agent-generated code.

    Uses an LLM to evaluate code quality, UX polish, product fit, and edge cases
    based on a provided rubric.
    """

    def __init__(self, model: str = "gemini/gemini-2.0-flash-001", batch_size: int = 5, step_delay: float = 0.0):
        """
        Initialize the rubric judge.

        Args:
            model: LiteLLM model identifier (default: gemini/gemini-2.0-flash-001)
            batch_size: Number of rubric items to evaluate per batch (default: 5)
            step_delay: Delay in seconds between batch evaluations.
                       Individual items within a batch are delayed by ceiling(step_delay/3) seconds.
                       (default: 0.0)
        """
        self.model = model
        self.batch_size = batch_size
        self.step_delay = step_delay

    def _discover_source_files(self, workspace_path: Path) -> List[Path]:
        """
        Recursively discover source code files in the workspace.

        Finds files with extensions: .ts, .tsx, .js, .jsx, .css

        Filters out:
        - node_modules/
        - .next/
        - .git/
        - Config files (package.json, pnpm-lock.yaml, tsconfig.json, etc.)

        Focuses on code the agent wrote (app/, components/, pages/, etc.)

        Args:
            workspace_path: Path to the workspace directory

        Returns:
            List of source file paths
        """
        source_extensions = {'.ts', '.tsx', '.js', '.jsx', '.css'}
        ignore_dirs = {'node_modules', '.next', '.git', 'dist', 'build', 'out'}
        ignore_files = {
            'package.json', 'package-lock.json', 'pnpm-lock.yaml',
            'tsconfig.json', 'next.config.js', 'next.config.ts',
            'tailwind.config.js', 'tailwind.config.ts',
            'postcss.config.js', 'postcss.config.mjs',
            'eslint.config.js', 'eslint.config.mjs',
            '.gitignore', '.eslintrc', '.prettierrc'
        }

        source_files = []

        for root, dirs, files in os.walk(workspace_path):
            # Filter out ignored directories in-place
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for file in files:
                # Check extension
                if Path(file).suffix not in source_extensions:
                    continue

                # Check if it's an ignored config file
                if file in ignore_files:
                    continue

                # Add the file
                file_path = Path(root) / file
                source_files.append(file_path)

        return source_files

    def _assemble_code_context(self, workspace_path: Path, source_files: List[Path]) -> str:
        """
        Assemble source code files into a single context string.

        Format:
        ```
        === path/to/file1.tsx ===
        <file contents>

        === path/to/file2.ts ===
        <file contents>
        ```

        Args:
            workspace_path: Base workspace path
            source_files: List of source file paths

        Returns:
            Concatenated code context string
        """
        context_parts = []

        for file_path in sorted(source_files):
            # Get relative path for cleaner display
            rel_path = file_path.relative_to(workspace_path)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                context_parts.append(f"=== {rel_path} ===\n{content}\n")
            except Exception as e:
                context_parts.append(f"=== {rel_path} ===\n[Error reading file: {e}]\n")

        return "\n".join(context_parts)

    def _parse_rubric(self, rubric_text: str) -> List[str]:
        """
        Parse a numbered rubric into individual requirement strings.

        Expects format like:
        1. Feature A
        2. Feature B
        3. Feature C

        Args:
            rubric_text: Rubric as a numbered list

        Returns:
            List of requirement strings (e.g., ["1. Feature A", "2. Feature B", ...])
        """
        # Split by lines and filter out empty lines
        lines = [line.strip() for line in rubric_text.strip().split('\n') if line.strip()]

        # Extract numbered items using regex
        # Matches: "1. Text", "1) Text", "1 - Text", etc.
        numbered_pattern = re.compile(r'^\d+[\.\)]\s*.+')

        rubric_items = []
        for line in lines:
            if numbered_pattern.match(line):
                rubric_items.append(line)

        return rubric_items

    def _load_system_log(self, workspace_path: Path) -> str:
        """
        Load system.log from the episode logs directory.

        Args:
            workspace_path: Path to the workspace directory

        Returns:
            Contents of system.log or empty string if not found
        """
        # System log is in runs/<timestamp>/logs/system.log
        # Workspace is runs/<timestamp>/workspace
        episode_dir = workspace_path.parent
        system_log_path = episode_dir / "logs" / "system.log"

        if not system_log_path.exists():
            return ""

        try:
            with open(system_log_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ Warning: Could not read system.log: {e}")
            return ""

    def _evaluate_single_item(
        self,
        rubric_item: str,
        code_context: str,
        system_log: str
    ) -> Dict:
        """
        Evaluate a single rubric item using evidence-based grading.

        Args:
            rubric_item: Single rubric requirement string
            code_context: Assembled source code
            system_log: Contents of system.log (build/install output)

        Returns:
            Evaluation result with format:
            {
                "item": "1. Feature description...",
                "status": "PASS" or "FAIL",
                "evidence": "file.ts:42 - description"
            }
        """
        system_prompt = """You are a Senior QA Engineer performing evidence-based code review.

Your task is to verify whether a specific requirement is implemented in the codebase.

CRITICAL RULES:
1. For PASS, you MUST cite the specific file path and line number where the requirement is implemented.
2. If you cannot find concrete evidence in the code, mark it as FAIL.
3. Do not hallucinate or assume code exists. Only cite code you can actually see.
4. Use system logs (build/install output) as supporting evidence when relevant.

You will be given:
- A requirement to verify
- The complete source code
- System logs (build, install, server output)

Determine:
- Status: PASS or FAIL
- Evidence: File path, line numbers, and brief explanation (for PASS) OR reason for failure (for FAIL)

CRITICAL: You must respond with valid JSON only. No markdown, no code blocks, just pure JSON.

Response format:
{
  "item": "1. Application implements user registration...",
  "status": "PASS",
  "evidence": "app/auth/route.ts:42-55 - POST endpoint with supabase.auth.signUp call"
}"""

        user_message = f"""# Requirement to Verify

{rubric_item}

# Source Code

{code_context}

# System Logs (Build/Install Output)

{system_log[:5000]}  # Truncate to first 5000 chars

Verify the requirement and respond with JSON only."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.2,  # Low temperature for factual verification
                max_tokens=1000,
                num_retries=3,
                timeout=60
            )

            response_text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or (not line.startswith("```") and "{" in line):
                        json_lines.append(line)
                response_text = "\n".join(json_lines)

            result = json.loads(response_text)

            # Validate structure
            if "status" not in result or "evidence" not in result:
                raise ValueError("LLM response missing 'status' or 'evidence'")

            # Ensure the item field is present
            if "item" not in result:
                result["item"] = rubric_item

            return result

        except Exception as e:
            print(f"❌ Error evaluating item: {e}")
            return {
                "item": rubric_item,
                "status": "FAIL",
                "evidence": f"Evaluation error: {str(e)}"
            }

    def _evaluate_batch(
        self,
        batch_items: List[str],
        code_context: str,
        system_log: str
    ) -> List[Dict]:
        """
        Evaluate a batch of rubric items using evidence-based grading.

        Args:
            batch_items: List of rubric requirement strings
            code_context: Assembled source code
            system_log: Contents of system.log (build/install output)

        Returns:
            List of evaluation results with format:
            [
                {
                    "item": "1. Feature description...",
                    "status": "PASS" or "FAIL",
                    "evidence": "file.ts:42 - description"
                },
                ...
            ]
        """
        results = []

        # Calculate delay between individual items
        item_delay = math.ceil(self.step_delay / 3) if self.step_delay > 0 else 0

        for idx, item in enumerate(batch_items):
            # Evaluate single item
            result = self._evaluate_single_item(
                rubric_item=item,
                code_context=code_context,
                system_log=system_log
            )
            results.append(result)

            # Add delay between items (but not after the last one in the batch)
            if item_delay > 0 and idx < len(batch_items) - 1:
                print(f"    ⏱️  Item delay: {item_delay}s...")
                time.sleep(item_delay)

        return results

    def _build_prompt(self, user_prompt: str, rubric: str, code_context: str) -> List[Dict]:
        """
        Build the LLM prompt messages.

        Args:
            user_prompt: Original task prompt given to the agent
            rubric: Grading rubric
            code_context: Assembled source code

        Returns:
            List of message dicts for LiteLLM
        """
        system_prompt = """You are a Senior QA Engineer and Code Reviewer evaluating a web application built by an AI agent.

Your task is to evaluate the code quality, functionality, and alignment with the requirements.

You will be given:
1. The original task prompt (what the agent was asked to build)
2. A grading rubric (evaluation criteria)
3. The source code the agent produced

Evaluate the code based on the rubric and provide a detailed assessment.

CRITICAL: You must respond with valid JSON only. No markdown, no code blocks, just pure JSON.

Response format:
{
  "score": <number 0-100>,
  "reasoning": "<detailed explanation of your evaluation>",
  "breakdown": {
    "<criterion1>": <score 0-100>,
    "<criterion2>": <score 0-100>,
    ...
  }
}"""

        user_message = f"""# Task Prompt
{user_prompt}

# Grading Rubric
{rubric}

# Source Code

{code_context}

Evaluate the code based on the rubric above. Respond with JSON only."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

    def evaluate(self, workspace_path: str, prompt: str, rubric: str, system_log: str = "") -> dict:
        """
        Evaluate agent-generated code using batched, evidence-based grading.

        Args:
            workspace_path: Path to the workspace directory
            prompt: Original task prompt given to the agent (not used in batched mode)
            rubric: Grading rubric as a numbered list
            system_log: System log contents (build/install output)

        Returns:
            Dict with:
            {
                "score": float (0.0-1.0, ratio of passed/total),
                "reasoning": str,
                "breakdown": [
                    {
                        "item": "1. Feature description...",
                        "status": "PASS" or "FAIL",
                        "evidence": "file.ts:42 - description"
                    },
                    ...
                ],
                "metadata": {
                    "files_evaluated": int,
                    "model": str,
                    "total_items": int,
                    "passed_items": int
                }
            }
        """
        workspace = Path(workspace_path).resolve()

        if not workspace.exists():
            raise ValueError(f"Workspace does not exist: {workspace_path}")

        print("🔍 Discovering source files...")
        source_files = self._discover_source_files(workspace)
        print(f"Found {len(source_files)} source files")

        if not source_files:
            print("⚠️ Warning: No source files found to evaluate")
            return {
                "score": 0.0,
                "reasoning": "No source files found in workspace",
                "breakdown": [],
                "metadata": {
                    "files_evaluated": 0,
                    "model": self.model,
                    "total_items": 0,
                    "passed_items": 0
                }
            }

        print("📝 Assembling code context...")
        code_context = self._assemble_code_context(workspace, source_files)

        # If system_log not provided, try to load it
        if not system_log:
            print("📋 Loading system logs...")
            system_log = self._load_system_log(workspace)

        print("🔢 Parsing rubric...")
        rubric_items = self._parse_rubric(rubric)
        print(f"Found {len(rubric_items)} requirements to evaluate")

        if not rubric_items:
            print("⚠️ Warning: No numbered requirements found in rubric")
            return {
                "score": 0.0,
                "reasoning": "No numbered requirements found in rubric",
                "breakdown": [],
                "metadata": {
                    "files_evaluated": len(source_files),
                    "model": self.model,
                    "total_items": 0,
                    "passed_items": 0
                }
            }

        # Split rubric into batches
        all_results = []
        num_batches = (len(rubric_items) + self.batch_size - 1) // self.batch_size

        print(f"🤖 Evaluating in {num_batches} batches of {self.batch_size}...")

        for i in range(0, len(rubric_items), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch_items = rubric_items[i:i + self.batch_size]

            print(f"  Batch {batch_num}/{num_batches}: Evaluating {len(batch_items)} items...")

            batch_results = self._evaluate_batch(
                batch_items=batch_items,
                code_context=code_context,
                system_log=system_log
            )

            all_results.extend(batch_results)

            print(f"  ✓ Batch {batch_num} complete")

            # Add delay between batches (but not after the last one)
            if self.step_delay > 0 and batch_num < num_batches:
                print(f"  ⏱️  Waiting {self.step_delay}s before next batch...")
                time.sleep(self.step_delay)

        # Calculate stats
        total_items = len(all_results)
        passed_items = sum(1 for item in all_results if item.get("status") == "PASS")

        # Score = ratio of passed items (0.0 to 1.0)
        score = passed_items / total_items if total_items > 0 else 0.0

        reasoning = (
            f"Evaluated {total_items} items in {num_batches} batches. "
            f"{passed_items}/{total_items} requirements passed ({score:.1%})."
        )

        print(f"✅ Evaluation complete: {passed_items}/{total_items} passed → Score: {score:.1%}")

        return {
            "score": score,
            "reasoning": reasoning,
            "breakdown": all_results,
            "metadata": {
                "files_evaluated": len(source_files),
                "model": self.model,
                "total_items": total_items,
                "passed_items": passed_items
            }
        }
