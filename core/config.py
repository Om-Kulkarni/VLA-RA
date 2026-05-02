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

    llm_model: str = "deepseek/deepseek-v3.2"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    max_papers_per_run: int = 5
    db_path: str = "data/vla_ra.db"


def get_config() -> ProjectConfig:
    """
    Retrieves global configuration values from environment variables and defaults.

    Returns:
        ProjectConfig: A dataclass instance containing configuration parameters.
    """
    return ProjectConfig(
        llm_model=os.getenv("LLM_MODEL", "deepseek/deepseek-v3.2"),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        max_papers_per_run=int(os.getenv("MAX_PAPERS_PER_RUN", 5)),
        db_path=os.getenv("DB_PATH", "data/vla_ra.db"),
    )
