"""
Grader module for evaluating Next.js application builds.

This module provides the Grader class which handles:
- Installing dependencies with robust architecture detection
- Building the application to verify it compiles
- Logging results for evaluation
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple


class Grader:
    """
    Grader for evaluating Next.js application builds.

    This class handles installing dependencies and building the application
    to verify it compiles without errors.
    """

    def __init__(self, workspace_dir: str):
        """
        Initialize the grader with a workspace directory.

        Args:
            workspace_dir: Path to the workspace directory containing the Next.js app
        """
        self.workspace_dir = Path(workspace_dir).resolve()

        if not self.workspace_dir.exists():
            raise ValueError(f"Workspace directory does not exist: {workspace_dir}")

    def _execute(
        self,
        command: str,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 600
    ) -> Tuple[bool, str, str, int]:
        """
        Execute a shell command safely.

        Merges custom environment variables with os.environ to preserve PATH
        and other system variables, just like our Sandbox class.

        Args:
            command: Command to execute
            env: Environment variables to add (merged with os.environ)
            timeout: Timeout in seconds (default: 600)

        Returns:
            Tuple of (success, stdout, stderr, exit_code)
        """
        # Merge custom env with system environment to preserve PATH
        full_env = os.environ.copy()
        if env is not None:
            full_env.update(env)

        try:
            result = subprocess.run(
                command,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                env=full_env
            )

            success = result.returncode == 0
            return success, result.stdout, result.stderr, result.returncode

        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if e.stdout else ""
            stderr = e.stderr.decode() if e.stderr else ""
            return False, stdout, stderr, -1

        except Exception as e:
            return False, "", str(e), -1

    def run_install(self) -> bool:
        """
        Install dependencies using the robust logic from env/tools.py.

        This method replicates the exact same logic we use in the agent's
        install_deps tool:
        1. Deletes node_modules, lockfiles to ensure clean install
        2. Detects Node.js architecture (not Python's, to avoid Rosetta issues)
        3. Sets npm_config_arch to match Node's architecture
        4. Runs pnpm install --no-frozen-lockfile
        5. Runs pnpm rebuild as safety net for native modules

        Returns:
            True if installation succeeded, False otherwise
        """
        try:
            # 1. CLEANUP: Delete node_modules for clean install
            print("Cleaning up old dependencies...")
            items_to_delete = ["pnpm-lock.yaml", "package-lock.json", "node_modules"]
            for item in items_to_delete:
                item_path = self.workspace_dir / item
                if item_path.exists():
                    try:
                        if item_path.is_dir():
                            shutil.rmtree(item_path)
                        else:
                            item_path.unlink()
                    except Exception:
                        pass

            # 2. DETECT NODE ARCHITECTURE
            # Ask Node directly what architecture it's running on
            # This bypasses Python Rosetta/Intel emulation issues
            print("Detecting Node.js architecture...")
            success, stdout, stderr, _ = self._execute('node -p "process.arch"')
            node_arch = stdout.strip() if success else "arm64"
            print(f"Target Architecture: {node_arch}")

            # 3. PREPARE ENVIRONMENT
            # Force pnpm to install for the NODE architecture, not Python's
            success, stdout, stderr, _ = self._execute("uname")
            platform = "darwin" if "darwin" in stdout.lower() else "linux"

            install_env = {
                "CI": "false",
                "npm_config_arch": node_arch,  # Force pnpm to use Node's arch
                "npm_config_platform": platform,
            }

            # 4. INSTALL (Standard pnpm)
            print(f"Installing dependencies with pnpm for {node_arch}...")
            success, stdout, stderr, exit_code = self._execute(
                "pnpm install --no-frozen-lockfile",
                env=install_env,
                timeout=600
            )

            if not success:
                print(f"❌ Install failed with exit code {exit_code}")
                print(f"STDERR: {stderr}")
                return False

            print(stdout)

            # 5. REBUILD (The Safety Net)
            print("🔧 Rebuilding native modules...")
            success, stdout, stderr, _ = self._execute(
                "pnpm rebuild",
                env=install_env,
                timeout=300
            )

            print("✅ Dependencies installed successfully")
            return True

        except Exception as e:
            print(f"❌ Error during installation: {str(e)}")
            return False

    def run_build(self) -> bool:
        """
        Run pnpm build to verify the application compiles.

        Runs 'pnpm build' and captures stdout/stderr for logging.
        Returns True only if exit code is 0 (success).

        Returns:
            True if build succeeded (exit code 0), False otherwise
        """
        try:
            print("🏗️ Building application...")
            success, stdout, stderr, exit_code = self._execute(
                "pnpm build",
                timeout=600
            )

            # Log output
            if stdout:
                print(stdout)

            if not success:
                print(f"❌ Build failed with exit code {exit_code}")
                if stderr:
                    print(f"STDERR: {stderr}")
                return False

            print("✅ Build succeeded")
            return True

        except Exception as e:
            print(f"❌ Error during build: {str(e)}")
            return False
