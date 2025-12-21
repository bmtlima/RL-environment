"""
Sandbox module for managing isolated working directories and executing commands.
"""

import subprocess
from pathlib import Path
from typing import Dict, Optional


class SandboxResult:
    """Result of a sandbox command execution."""

    def __init__(
        self,
        success: bool,
        stdout: str,
        stderr: str,
        exit_code: int,
        error: Optional[str] = None,
        timed_out: bool = False
    ):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.error = error
        self.timed_out = timed_out

    def to_dict(self) -> Dict[str, any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "error": self.error,
            "timed_out": self.timed_out
        }


class Sandbox:
    """
    Manages a working directory and provides safe command execution.

    This class handles:
    - Working directory management
    - Safe command execution with timeout
    - stdout/stderr capture
    - Error handling without crashes
    """

    def __init__(self, workspace_dir: Path, default_timeout: int = 300):
        """
        Initialize a sandbox with a workspace directory.

        Args:
            workspace_dir: Path to the workspace directory
            default_timeout: Default timeout for commands in seconds (default: 300s/5min)
        """
        self.workspace_dir = Path(workspace_dir).resolve()
        self.default_timeout = default_timeout

        # Create workspace directory if it doesn't exist
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        shell: bool = True
    ) -> SandboxResult:
        """
        Execute a shell command in the sandbox.

        Args:
            command: Command to execute
            cwd: Working directory (defaults to workspace_dir)
            timeout: Timeout in seconds (defaults to default_timeout)
            env: Environment variables to pass to the command
            shell: Whether to execute through shell (default: True)

        Returns:
            SandboxResult object with execution results
        """
        # Use workspace_dir if cwd is not specified
        if cwd is None:
            cwd = self.workspace_dir
        else:
            cwd = Path(cwd).resolve()

            # Ensure cwd is within workspace (security check)
            try:
                cwd.relative_to(self.workspace_dir)
            except ValueError:
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    error=f"cwd must be within workspace directory: {self.workspace_dir}"
                )

        # Create cwd if it doesn't exist
        cwd.mkdir(parents=True, exist_ok=True)

        # Use default timeout if not specified
        if timeout is None:
            timeout = self.default_timeout

        try:
            # Execute command
            result = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=shell,
                env=env
            )

            return SandboxResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode
            )

        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                success=False,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=e.stderr.decode() if e.stderr else "",
                exit_code=-1,
                error=f"Command timed out after {timeout} seconds",
                timed_out=True
            )

        except FileNotFoundError as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error=f"Command not found: {e}"
            )

        except PermissionError as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error=f"Permission denied: {e}"
            )

        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error=f"Unexpected error: {str(e)}"
            )

    def get_workspace_dir(self) -> Path:
        """Get the workspace directory path."""
        return self.workspace_dir

    def exists(self, path: Path) -> bool:
        """Check if a path exists within the workspace."""
        try:
            full_path = (self.workspace_dir / path).resolve()
            full_path.relative_to(self.workspace_dir)
            return full_path.exists()
        except (ValueError, Exception):
            return False

    def cleanup(self) -> None:
        """Clean up the sandbox workspace (placeholder for future implementation)."""
        # For now, we don't auto-delete. This could be extended to:
        # - Delete workspace directory
        # - Kill any running processes
        # - Clean up resources
        pass
