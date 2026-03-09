"""
Configuration Options

This module handles environment variables, API keys, and global configuration values.
"""

import os
from dataclasses import dataclass


@dataclass
class ProjectConfig:
    """
    Holds global configuration options for the VLA-RA project.
    """

    llm_model: str = "gemini-3.1-flash-lite-preview"
    max_papers_per_run: int = 5


def get_config() -> ProjectConfig:
    """
    Retrieves global configuration values from environment variables and defaults.

    Returns:
        ProjectConfig: A dataclass instance containing configuration parameters.
    """
    return ProjectConfig(
        # Use env var if present; otherwise default set in dataclass
        llm_model=os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview"),
        max_papers_per_run=int(os.getenv("MAX_PAPERS_PER_RUN", 5)),
    )
