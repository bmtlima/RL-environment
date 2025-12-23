"""
Test script for the RubricJudge module.

This script tests the LLM judge on a workspace with a dummy prompt and rubric.

Usage:
    python3 test_judge.py                    # Test template
    python3 test_judge.py path/to/workspace  # Test specific workspace
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from grader.rubric_judge import RubricJudge


def test_judge(workspace_path: str):
    """
    Test the rubric judge on a workspace.

    Args:
        workspace_path: Path to the workspace to evaluate
    """
    print(f"Testing RubricJudge on workspace: {workspace_path}\n")

    # Dummy task prompt
    task_prompt = """Build a simple counter application with the following features:
- Display a number starting at 0
- Button to increment the counter
- Button to decrement the counter
- Clean, modern UI using Tailwind CSS"""

    # Dummy grading rubric
    rubric = """Evaluate the code on the following criteria (each 0-100):

1. Functionality (40 points)
   - Counter displays correctly
   - Increment button works
   - Decrement button works
   - State management is correct

2. Code Quality (30 points)
   - Clean, readable code
   - Proper TypeScript types
   - Good component structure
   - No unnecessary complexity

3. UI/UX (20 points)
   - Clean, modern design
   - Good use of Tailwind CSS
   - Responsive layout
   - Accessible buttons

4. Best Practices (10 points)
   - Follows React best practices
   - Proper use of hooks
   - No console errors
   - Production-ready code"""

    # Initialize judge
    try:
        judge = RubricJudge(model="gpt-4o-mini")
        print(f"✓ RubricJudge initialized with model: {judge.model}\n")
    except Exception as e:
        print(f"✗ Failed to initialize judge: {e}")
        return False

    # Run evaluation
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print()

    result = judge.evaluate(workspace_path, task_prompt, rubric)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Score: {result['score']}/100")
    print(f"\nReasoning:\n{result['reasoning']}")

    if result.get('breakdown'):
        print("\nBreakdown:")
        for criterion, score in result['breakdown'].items():
            print(f"  - {criterion}: {score}")

    print(f"\nMetadata:")
    print(f"  - Files evaluated: {result['metadata']['files_evaluated']}")
    print(f"  - Model: {result['metadata']['model']}")

    return result['score'] > 0


if __name__ == "__main__":
    # Default to testing the template
    workspace = "templates/nextjs-starter"

    # Allow user to specify workspace via command line
    if len(sys.argv) > 1:
        workspace = sys.argv[1]

    success = test_judge(workspace)
    sys.exit(0 if success else 1)
