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

    def check_server_health(self, port: int = 3000, timeout: int = 30) -> bool:
        """
        Start the production server and verify it responds with HTTP 200.

        This method:
        1. Starts 'pnpm start' (production server) using Popen
        2. Waits up to 30 seconds for the server to accept connections
        3. Sends HTTP GET to http://localhost:3000/
        4. Verifies response is 200 OK
        5. ALWAYS kills the server process and children (even on failure)

        Args:
            port: Port to check (default: 3000)
            timeout: Max seconds to wait for server (default: 30)

        Returns:
            True if server started and returned 200, False otherwise
        """
        import socket
        import time
        import signal

        try:
            import requests
        except ImportError:
            print("❌ requests library not found. Install with: pip install requests")
            return False

        process = None

        try:
            print(f"🚀 Starting production server on port {port}...")

            # Start server in background using Popen
            # preexec_fn=os.setsid creates a new process group for easy cleanup
            process = subprocess.Popen(
                "pnpm start",
                cwd=str(self.workspace_dir),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Create new process group
            )

            # Wait for server to be ready (socket check)
            print(f"⏳ Waiting for server to accept connections (max {timeout}s)...")
            start_time = time.time()
            server_ready = False

            while time.time() - start_time < timeout:
                try:
                    # Try to connect to the port
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()

                    if result == 0:
                        server_ready = True
                        break
                except Exception:
                    pass

                # Check if process died
                if process.poll() is not None:
                    print("❌ Server process died during startup")
                    stdout, stderr = process.communicate()
                    if stdout:
                        print(f"STDOUT: {stdout}")
                    if stderr:
                        print(f"STDERR: {stderr}")
                    return False

                time.sleep(0.5)

            if not server_ready:
                print(f"❌ Server did not start within {timeout} seconds")
                return False

            print("✓ Server is accepting connections")

            # Verify HTTP 200 response
            print(f"📡 Sending HTTP GET to http://localhost:{port}/...")
            response = requests.get(f"http://localhost:{port}/", timeout=5)

            if response.status_code == 200:
                print(f"✅ Server health check passed (HTTP {response.status_code})")
                return True
            else:
                print(f"❌ Server returned HTTP {response.status_code} (expected 200)")
                return False

        except requests.RequestException as e:
            print(f"❌ HTTP request failed: {e}")
            return False

        except Exception as e:
            print(f"❌ Error during server health check: {e}")
            return False

        finally:
            # CRITICAL: Always kill the server process
            if process is not None:
                print("🛑 Shutting down server...")
                try:
                    # Kill the entire process group to catch child processes
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't die gracefully
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait()

                    print("✓ Server shut down successfully")

                except Exception as e:
                    print(f"⚠️ Error during cleanup: {e}")
                    # Last resort: try to kill directly
                    try:
                        process.kill()
                        process.wait()
                    except Exception:
                        pass

    def run_all_checks(self) -> dict:
        """
        Run all grading checks in sequence.

        Executes:
        1. run_install() - Install dependencies
        2. run_build() - Build production bundle
        3. check_server_health() - Start and verify server (only if build passes)

        Returns:
            Dictionary with results:
            {
                "install": bool,
                "build": bool,
                "server_health": bool,
                "overall_pass": bool
            }
        """
        results = {
            "install": False,
            "build": False,
            "server_health": False,
            "overall_pass": False
        }

        print("=" * 60)
        print("GRADING: Running All Checks")
        print("=" * 60)
        print()

        # Check 1: Install
        print("CHECK 1/3: Installing Dependencies")
        print("-" * 60)
        results["install"] = self.run_install()
        print()

        if not results["install"]:
            print("❌ Install failed - skipping remaining checks")
            return results

        # Check 2: Build
        print("CHECK 2/3: Building Application")
        print("-" * 60)
        results["build"] = self.run_build()
        print()

        if not results["build"]:
            print("❌ Build failed - skipping server health check")
            return results

        # Check 3: Server Health
        print("CHECK 3/3: Server Health Check")
        print("-" * 60)
        results["server_health"] = self.check_server_health()
        print()

        # Overall pass requires all checks to pass
        results["overall_pass"] = all([
            results["install"],
            results["build"],
            results["server_health"]
        ])

        # Summary
        print("=" * 60)
        print("GRADING SUMMARY")
        print("=" * 60)
        print(f"Install:       {'✅ PASS' if results['install'] else '❌ FAIL'}")
        print(f"Build:         {'✅ PASS' if results['build'] else '❌ FAIL'}")
        print(f"Server Health: {'✅ PASS' if results['server_health'] else '❌ FAIL'}")
        print(f"Overall:       {'✅ PASS' if results['overall_pass'] else '❌ FAIL'}")
        print()

        return results
