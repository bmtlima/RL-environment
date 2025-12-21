"""
LLM interface using LiteLLM for model-agnostic API calls.

This module provides a simple interface to call various LLM providers
(OpenAI, Anthropic, OpenRouter, etc.) through LiteLLM.

Environment Variables Required:
- OPENAI_API_KEY: For OpenAI models (gpt-4o, gpt-4o-mini, etc.)
- ANTHROPIC_API_KEY: For Anthropic models (claude-3-5-sonnet, etc.)
- OPENROUTER_API_KEY: For OpenRouter models
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import litellm
from litellm import completion


class ModelConfig:
    """Configuration for a specific LLM model."""

    def __init__(
        self,
        model_id: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        description: str = "",
        env_var: str = ""
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.description = description
        self.env_var = env_var

    def __repr__(self) -> str:
        return f"ModelConfig(id={self.model_id}, name={self.model_name}, temp={self.temperature})"


class LLMInterface:
    """
    Interface for calling LLMs via LiteLLM.

    This class handles:
    - Loading model configurations from YAML
    - Making API calls to various LLM providers
    - Error handling and response parsing
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the LLM interface.

        Args:
            config_path: Path to models.yaml config file (defaults to configs/models.yaml)
        """
        if config_path is None:
            # Default to configs/models.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "configs" / "models.yaml"

        self.config_path = Path(config_path)
        self.models: Dict[str, ModelConfig] = {}
        self.default_model: str = ""

        # Load configuration
        self._load_config()

        # Configure LiteLLM
        litellm.drop_params = True  # Drop unsupported params instead of erroring
        litellm.set_verbose = False  # Disable verbose logging by default

    def _load_config(self) -> None:
        """Load model configurations from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Parse model configurations
        models_config = config.get('models', {})
        for model_id, model_data in models_config.items():
            self.models[model_id] = ModelConfig(
                model_id=model_id,
                model_name=model_data['model_name'],
                temperature=model_data.get('temperature', 0.7),
                max_tokens=model_data.get('max_tokens', 4096),
                description=model_data.get('description', ''),
                env_var=model_data.get('env_var', '')
            )

        # Set default model
        self.default_model = config.get('default_model', list(self.models.keys())[0] if self.models else '')

    def get_model_config(self, model_id: Optional[str] = None) -> ModelConfig:
        """
        Get configuration for a specific model.

        Args:
            model_id: Model identifier (uses default if not specified)

        Returns:
            ModelConfig object

        Raises:
            ValueError: If model_id is not found in config
        """
        if model_id is None:
            model_id = self.default_model

        if model_id not in self.models:
            available = ', '.join(self.models.keys())
            raise ValueError(f"Model '{model_id}' not found in config. Available: {available}")

        return self.models[model_id]

    def call_model(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Call an LLM with the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Example: [{"role": "user", "content": "Hello!"}]
            model_id: Model identifier (uses default if not specified)
            temperature: Override config temperature (optional)
            max_tokens: Override config max_tokens (optional)
            **kwargs: Additional parameters to pass to LiteLLM

        Returns:
            String response from the model

        Raises:
            ValueError: If model configuration is invalid
            Exception: If API call fails
        """
        # Get model configuration
        config = self.get_model_config(model_id)

        # Check if API key is set
        if config.env_var and not os.getenv(config.env_var):
            raise ValueError(
                f"Environment variable '{config.env_var}' not set for model '{config.model_id}'. "
                f"Please set it in your .env file or environment."
            )

        # Prepare parameters
        params = {
            "model": config.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
            **kwargs
        }

        try:
            # Make API call via LiteLLM
            response = completion(**params)

            # Extract content from response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise Exception("No response choices returned from model")

        except Exception as e:
            raise Exception(f"Error calling model '{config.model_id}': {str(e)}")

    def list_models(self) -> List[str]:
        """
        Get list of available model IDs.

        Returns:
            List of model identifier strings
        """
        return list(self.models.keys())

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model information
        """
        config = self.get_model_config(model_id)
        return {
            "id": config.model_id,
            "name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "description": config.description,
            "env_var": config.env_var,
            "env_var_set": bool(os.getenv(config.env_var)) if config.env_var else None
        }


# Convenience function for simple usage
def call_model(
    messages: List[Dict[str, str]],
    model_id: Optional[str] = None,
    config_path: Optional[Path] = None,
    **kwargs
) -> str:
    """
    Convenience function to call a model without managing LLMInterface instance.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_id: Model identifier (uses default if not specified)
        config_path: Path to models.yaml config file (optional)
        **kwargs: Additional parameters to pass to LiteLLM

    Returns:
        String response from the model
    """
    llm = LLMInterface(config_path=config_path)
    return llm.call_model(messages=messages, model_id=model_id, **kwargs)
