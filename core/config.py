"""
Configuration Options

This module handles environment variables, API keys, and global configuration values.
"""
import os

def get_config() -> dict:
    """
    Retrieves global configuration values from environment variables.
    
    Returns:
        dict: A dictionary of configuration keys and limits.
    """
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
    }
