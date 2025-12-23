"""
Rubric-based LLM judge for evaluating agent-generated code.

This module provides subjective evaluation of code quality, UX polish,
product fit, and edge cases using an LLM as a judge.
"""

import os
import json
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

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the rubric judge.

        Args:
            model: LiteLLM model identifier (default: gpt-4o-mini)
        """
        self.model = model

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

    def evaluate(self, workspace_path: str, prompt: str, rubric: str) -> dict:
        """
        Evaluate agent-generated code using an LLM judge.

        Args:
            workspace_path: Path to the workspace directory
            prompt: Original task prompt given to the agent
            rubric: Grading rubric (criteria for evaluation)

        Returns:
            Dict with:
            {
                "score": int (0-100),
                "reasoning": str,
                "breakdown": dict,
                "metadata": {
                    "files_evaluated": int,
                    "model": str
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
                "score": 0,
                "reasoning": "No source files found in workspace",
                "breakdown": {},
                "metadata": {
                    "files_evaluated": 0,
                    "model": self.model
                }
            }

        print("📝 Assembling code context...")
        code_context = self._assemble_code_context(workspace, source_files)

        print(f"🤖 Calling LLM judge ({self.model})...")
        messages = self._build_prompt(prompt, rubric, code_context)

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent evaluation
                max_tokens=2000
            )

            response_text = response.choices[0].message.content.strip()

            # Try to parse JSON
            # Handle case where LLM wraps JSON in markdown code blocks
            if response_text.startswith("```"):
                # Extract JSON from code block
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
            if "score" not in result or "reasoning" not in result:
                raise ValueError("LLM response missing required fields")

            # Add metadata
            result["metadata"] = {
                "files_evaluated": len(source_files),
                "model": self.model
            }

            print(f"✅ Evaluation complete (Score: {result['score']}/100)")
            return result

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {response_text}")
            return {
                "score": 0,
                "reasoning": f"Failed to parse LLM response: {e}",
                "breakdown": {},
                "metadata": {
                    "files_evaluated": len(source_files),
                    "model": self.model,
                    "error": "json_parse_error"
                }
            }
        except Exception as e:
            print(f"❌ Error during LLM evaluation: {e}")
            return {
                "score": 0,
                "reasoning": f"Error during evaluation: {e}",
                "breakdown": {},
                "metadata": {
                    "files_evaluated": len(source_files),
                    "model": self.model,
                    "error": str(e)
                }
            }
