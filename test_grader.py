"""
Test script for the Grader module.

This script verifies that the Grader can:
1. Install dependencies in a Next.js workspace
2. Build the application successfully
3. Start the production server and verify HTTP 200

Usage:
    python test_grader.py                    # Test template
    python test_grader.py path/to/workspace  # Test specific workspace
    python test_grader.py --all              # Run all checks including server health
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from grader import Grader


def test_grader_basic(workspace_path: str):
    """
    Test install and build only (faster).

    Args:
        workspace_path: Path to the workspace to test
    """
    print(f"Testing Grader (basic) on workspace: {workspace_path}\n")

    # Initialize grader
    try:
        grader = Grader(workspace_path)
        print(f"✓ Grader initialized with workspace: {grader.workspace_dir}\n")
    except Exception as e:
        print(f"✗ Failed to initialize grader: {e}")
        return False

    # Test run_install()
    print("=" * 60)
    print("TEST 1: run_install()")
    print("=" * 60)
    install_success = grader.run_install()
    print()

    if not install_success:
        print("✗ Install failed - stopping tests")
        return False

    print("✓ Install passed\n")

    # Test run_build()
    print("=" * 60)
    print("TEST 2: run_build()")
    print("=" * 60)
    build_success = grader.run_build()
    print()

    if not build_success:
        print("✗ Build failed")
        return False

    print("✓ Build passed\n")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Install: {'✓ PASS' if install_success else '✗ FAIL'}")
    print(f"Build:   {'✓ PASS' if build_success else '✗ FAIL'}")
    print()

    return install_success and build_success


def test_grader_full(workspace_path: str):
    """
    Test all checks including server health (slower).

    Args:
        workspace_path: Path to the workspace to test
    """
    print(f"Testing Grader (full) on workspace: {workspace_path}\n")

    # Initialize grader
    try:
        grader = Grader(workspace_path)
        print(f"✓ Grader initialized with workspace: {grader.workspace_dir}\n")
    except Exception as e:
        print(f"✗ Failed to initialize grader: {e}")
        return False

    # Run all checks
    results = grader.run_all_checks()

    # Return overall pass/fail
    return results["overall_pass"]


if __name__ == "__main__":
    # Default to testing the template
    workspace = "templates/nextjs-starter"
    run_all = False

    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            run_all = True
        else:
            workspace = sys.argv[1]
            if len(sys.argv) > 2 and sys.argv[2] == "--all":
                run_all = True

    # Run appropriate test
    if run_all:
        success = test_grader_full(workspace)
    else:
        success = test_grader_basic(workspace)

    sys.exit(0 if success else 1)
