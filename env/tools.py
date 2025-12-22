"""
Tools module providing file operations and command execution via the Sandbox.
"""

from pathlib import Path
from typing import Dict, Optional, Union
from env.sandbox import Sandbox


class ToolResult:
    """Result of a tool operation."""

    def __init__(
        self,
        success: bool,
        data: Optional[any] = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.data = data
        self.error = error

    def to_dict(self) -> Dict[str, any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error
        }


class Tools:
    """
    Provides file and command execution tools using a Sandbox.

    This class exposes tools that agents can use to interact with the filesystem
    and execute commands within the sandboxed workspace.
    """

    def __init__(self, sandbox: Sandbox):
        """
        Initialize tools with a sandbox instance.

        Args:
            sandbox: Sandbox instance to use for operations
        """
        self.sandbox = sandbox

    def write_file(self, path: str, content: str) -> ToolResult:
        """
        Write content to a file within the sandbox workspace.

        Args:
            path: Relative path to the file (relative to workspace)
            content: Content to write to the file

        Returns:
            ToolResult indicating success or failure
        """
        try:
            # Resolve path within workspace
            file_path = (self.sandbox.workspace_dir / path).resolve()

            # Security check: ensure path is within workspace
            try:
                file_path.relative_to(self.sandbox.workspace_dir)
            except ValueError:
                return ToolResult(
                    success=False,
                    error=f"Path must be within workspace: {path}"
                )

            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content to file
            file_path.write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                data={"path": str(file_path.relative_to(self.sandbox.workspace_dir)), "bytes_written": len(content)}
            )

        except PermissionError as e:
            return ToolResult(
                success=False,
                error=f"Permission denied writing to {path}: {e}"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error writing file {path}: {str(e)}"
            )

    def read_file(self, path: str) -> ToolResult:
        """
        Read content from a file within the sandbox workspace.

        Args:
            path: Relative path to the file (relative to workspace)

        Returns:
            ToolResult with file content or error
        """
        try:
            # Resolve path within workspace
            file_path = (self.sandbox.workspace_dir / path).resolve()

            # Security check: ensure path is within workspace
            try:
                file_path.relative_to(self.sandbox.workspace_dir)
            except ValueError:
                return ToolResult(
                    success=False,
                    error=f"Path must be within workspace: {path}"
                )

            # Check if file exists
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}"
                )

            # Check if path is a file (not a directory)
            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    error=f"Path is not a file: {path}"
                )

            # Read file content
            content = file_path.read_text(encoding="utf-8")

            return ToolResult(
                success=True,
                data={"path": str(file_path.relative_to(self.sandbox.workspace_dir)), "content": content}
            )

        except PermissionError as e:
            return ToolResult(
                success=False,
                error=f"Permission denied reading {path}: {e}"
            )

        except UnicodeDecodeError as e:
            return ToolResult(
                success=False,
                error=f"Cannot read {path} as text file (binary file?): {e}"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error reading file {path}: {str(e)}"
            )

    def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> ToolResult:
        """
        Execute a shell command within the sandbox.

        Args:
            command: Command to execute
            cwd: Working directory relative to workspace (optional)
            timeout: Timeout in seconds (optional)

        Returns:
            ToolResult with command output or error
        """
        try:
            # Resolve cwd if provided
            working_dir = None
            if cwd:
                working_dir = (self.sandbox.workspace_dir / cwd).resolve()

                # Security check: ensure cwd is within workspace
                try:
                    working_dir.relative_to(self.sandbox.workspace_dir)
                except ValueError:
                    return ToolResult(
                        success=False,
                        error=f"Working directory must be within workspace: {cwd}"
                    )

            # Execute command
            result = self.sandbox.execute(
                command=command,
                cwd=working_dir,
                timeout=timeout
            )

            # Convert SandboxResult to ToolResult
            if result.success:
                return ToolResult(
                    success=True,
                    data={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "exit_code": result.exit_code
                    }
                )
            else:
                error_msg = result.error if result.error else f"Command failed with exit code {result.exit_code}"
                if result.stderr:
                    error_msg += f"\nStderr: {result.stderr}"

                return ToolResult(
                    success=False,
                    error=error_msg,
                    data={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "exit_code": result.exit_code,
                        "timed_out": result.timed_out
                    }
                )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error executing command: {str(e)}"
            )

    def list_files(self, path: str = ".", pattern: str = "*") -> ToolResult:
        """
        List files and directories within the sandbox workspace.

        Args:
            path: Relative path to list (relative to workspace, defaults to workspace root)
            pattern: Glob pattern to filter results (default: "*" for all files)

        Returns:
            ToolResult with list of files/directories or error
        """
        try:
            # Resolve path within workspace
            dir_path = (self.sandbox.workspace_dir / path).resolve()

            # Security check: ensure path is within workspace
            try:
                dir_path.relative_to(self.sandbox.workspace_dir)
            except ValueError:
                return ToolResult(
                    success=False,
                    error=f"Path must be within workspace: {path}"
                )

            # Check if path exists
            if not dir_path.exists():
                return ToolResult(
                    success=False,
                    error=f"Path not found: {path}"
                )

            # Check if path is a directory
            if not dir_path.is_dir():
                return ToolResult(
                    success=False,
                    error=f"Path is not a directory: {path}"
                )

            # List files using glob pattern
            files = []
            for item in sorted(dir_path.glob(pattern)):
                relative_path = item.relative_to(self.sandbox.workspace_dir)
                files.append({
                    "name": item.name,
                    "path": str(relative_path),
                    "is_file": item.is_file(),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else None
                })

            return ToolResult(
                success=True,
                data={
                    "path": str(dir_path.relative_to(self.sandbox.workspace_dir)) if path != "." else ".",
                    "files": files,
                    "count": len(files)
                }
            )

        except PermissionError as e:
            return ToolResult(
                success=False,
                error=f"Permission denied listing {path}: {e}"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error listing files in {path}: {str(e)}"
            )

    def finish_task(self, summary: str = "") -> ToolResult:
        """
        Signal that the agent has completed the task.

        This is a special tool that doesn't perform any action in the sandbox,
        but signals to the agent loop that the task is complete.

        Args:
            summary: Optional summary of what was accomplished

        Returns:
            ToolResult indicating success
        """
        return ToolResult(
            success=True,
            data={
                "summary": summary,
                "message": "Task marked as complete"
            }
        )

    def start_server(self, port: int = 3000) -> ToolResult:
        """
        Start the development server in the background.

        This starts 'pnpm dev' without blocking. The server will continue
        running until the sandbox is cleaned up.

        Args:
            port: Port to run the server on (default: 3000)

        Returns:
            ToolResult with server information
        """
        try:
            # Start server in background
            result = self.sandbox.run_background("pnpm dev")

            if result.success:
                return ToolResult(
                    success=True,
                    data={
                        "url": f"http://localhost:{port}",
                        "message": f"Development server started in background. Access at http://localhost:{port}",
                        "pid": result.stdout
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"Failed to start server: {result.error}"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error starting server: {str(e)}"
            )


# Standalone function interface for backward compatibility / simpler usage
def create_tools(workspace_dir: Union[str, Path]) -> Tools:
    """
    Create a Tools instance with a new Sandbox.

    Args:
        workspace_dir: Path to the workspace directory

    Returns:
        Tools instance
    """
    sandbox = Sandbox(Path(workspace_dir))
    return Tools(sandbox)
