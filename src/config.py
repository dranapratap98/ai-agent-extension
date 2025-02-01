from typing import Dict, Any, Optional
from pydantic import BaseModel
from pathlib import Path
import yaml
import os

class AgentConfig(BaseModel):
    """Configuration for the AI agent."""
    workspace_dir: str = "./workspace"
    default_language: str = "python"
    providers: Dict[str, Dict[str, Any]] = {
        "openai": {
            "api_key": "",
            "model": "gpt-4",
            "temperature": 0.7
        },
        "anthropic": {
            "api_key": "",
            "model": "claude-3-opus-20240229",
            "temperature": 0.7
        }
    }
    max_tokens: int = 2000
    logging_level: str = "INFO"

class ConfigManager:
    """Manages the configuration for the AI agent."""