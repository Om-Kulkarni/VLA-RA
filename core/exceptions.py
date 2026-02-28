"""
Custom Exceptions

This module defines project-specific exceptions to enforce the 'Fail Loudly' policy.
"""

class VLARAException(Exception):
    """Base exception for all VLA-RA specific errors."""
    pass

class PaperAlreadyProcessedError(VLARAException):
    """Raised when a paper has already been processed and exists in the database."""
    pass

class InsufficientDataForScoreError(VLARAException):
    """Raised when the LLM or parser fails to provide enough data to calculate a rubric score."""
    pass

class PDFParsingError(VLARAException):
    """Raised when Docling fails to parse a PDF document."""
    pass

class LLMScoringError(VLARAException):
    """Raised when an LLM fails to process or generate a valid score."""
    pass
