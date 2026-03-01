"""
Base Tool Abstraction

This module defines the BaseTool interface that all other tools must implement
to adhere to the Liskov Substitution Principle.
"""
from abc import ABC, abstractmethod
from typing import Any

class BaseTool(ABC):
    """
    Abstract base class for all tools in the VLA-RA framework.
    """
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Executes the primary logic of the tool.
        """
        pass
