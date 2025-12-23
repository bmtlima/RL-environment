"""
Utility to load environment variables from configs/env.yaml.

This replaces the use of python-dotenv and .env files, loading API keys
from a YAML configuration file instead.
"""

import os
import yaml
from pathlib import Path


def load_env_from_yaml(env_path: Path = None) -> None:
    """
    Load environment variables from configs/env.yaml.

    This function reads the YAML file and sets environment variables
    with uppercase keys (e.g., openai_api_key -> OPENAI_API_KEY).

    Args:
        env_path: Path to env.yaml file. If None, uses configs/env.yaml
                  relative to project root.

    Raises:
        FileNotFoundError: If env.yaml doesn't exist
        yaml.YAMLError: If env.yaml is invalid
    """
    if env_path is None:
        # Default to configs/env.yaml relative to this file
        project_root = Path(__file__).parent.parent
        env_path = project_root / "configs" / "env.yaml"

    env_path = Path(env_path)

    if not env_path.exists():
        # Try to provide helpful error message
        example_path = env_path.parent / "env.yaml.example"
        error_msg = f"Environment config not found: {env_path}\n\n"

        if example_path.exists():
            error_msg += (
                f"Please copy {example_path} to {env_path}\n"
                f"and add your API keys.\n\n"
                f"  cp {example_path} {env_path}\n"
            )
        else:
            error_msg += "Please create this file with your API keys."

        raise FileNotFoundError(error_msg)

    # Load YAML file
    with open(env_path, 'r') as f:
        env_vars = yaml.safe_load(f)

    if not env_vars:
        raise ValueError(f"No environment variables found in {env_path}")

    # Set environment variables
    # Convert keys to uppercase (openai_api_key -> OPENAI_API_KEY)
    for key, value in env_vars.items():
        if value and not isinstance(value, (dict, list)):
            env_key = key.upper()
            os.environ[env_key] = str(value)


# For backward compatibility, also support loading from .env if env.yaml doesn't exist
def load_env(env_path: Path = None) -> None:
    """
    Load environment variables, trying env.yaml first, then .env as fallback.

    Args:
        env_path: Path to env.yaml file (optional)
    """
    try:
        load_env_from_yaml(env_path)
    except FileNotFoundError:
        # Fallback to .env if env.yaml doesn't exist
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            raise FileNotFoundError(
                "Neither configs/env.yaml nor python-dotenv is available. "
                "Please create configs/env.yaml with your API keys."
            )
