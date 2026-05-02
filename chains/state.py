"""
Graph State Definitions

This module defines the TypedDict representations for the shared LangGraph state.
"""

from typing import TypedDict, List, Dict, Any


class GraphState(TypedDict):
    """
    Represents the complete state of the workflow graph passed between agent nodes.

    Attributes:
        search_query (str): Optional one-off narrowing hint for the Scout. Leave empty
            to let the manifesto fully determine research direction. Do NOT hardcode
            domain-specific topics here \u2014 edit core/manifesto.md instead.
        manifesto (str): The central research manifesto (core/manifesto.md). This is the
            SINGLE SOURCE OF TRUTH for all research direction across the entire workflow.
        scout_results (List[Dict[str, Any]]): Raw results returned by the Scout node.
        filtered_results (List[Dict[str, Any]]): Deduplicated results after Librarian processing.
        analysis_outputs (Dict[str, Any]): Multimodal analysis extractions from the Analyst node.
        rubric_scores (Dict[str, float]): Final scores computed by the Critic node.
        errors (List[str]): Any blocking errors encountered that corrupt graph processing.
    """

    search_query: str
    manifesto: str
    scout_results: List[Dict[str, Any]]
    filtered_results: List[Dict[str, Any]]
    approved_papers: List[str]
    analysis_outputs: Dict[str, Any]
    rubric_scores: Dict[str, float]
    errors: List[str]
