"""
Scoring Logic

This module provides the deterministic math and weighted rubric calculations
for evaluating papers and repositories. It operates on LLM-generated raw inputs.

According to O (Open/Closed principle), new scoring criteria should be added here
without modifying the Agent graph logic.
"""

from typing import Dict, Any


def calculate_rubric_score(criteria_inputs: Dict[str, Any]) -> float:
    """
    Calculates the final deterministic score based on a weighted rubric.

    Args:
        criteria_inputs (Dict[str, Any]): A dictionary of raw scores or categorical values
                                          provided by the Critic node LLM.
                                          Expected keys:
                                          - 'citations': int
                                          - 'has_elite_lab': bool
                                          - 'has_core_conference': bool
                                          - 'relevance_score': float (0.0 to 5.0)
                                          - 'has_code_links': bool
                                          - 'has_project_website': bool
                                          - 'is_continuation': bool

    Returns:
        float: The final weighted score for the paper or repository, normalized to 0-5.
    """
    score = 0.0

    # 1. Citations (Max 1.0)
    citations = criteria_inputs.get("citations", 0)
    if citations >= 50:
        score += 1.0
    elif citations >= 10:
        score += 0.5
    elif citations > 0:
        score += 0.2

    # 2. Lab Impact & Target Conferences (Max 1.0)
    lab_conf_score = 0.0
    if criteria_inputs.get("has_elite_lab", False):
        lab_conf_score += 0.5
    if criteria_inputs.get("has_core_conference", False):
        lab_conf_score += 0.5
    score += lab_conf_score

    # 3. Novelty & Personal Relevance (Max 1.5)
    relevance = criteria_inputs.get("relevance_score", 0.0)
    # Normalize the 0-5 LLM score to max 1.5 weight
    score += (max(0.0, min(5.0, float(relevance))) / 5.0) * 1.5

    # 4. Reproducibility / Code Links (Max 1.0)
    if criteria_inputs.get("has_code_links", False):
        score += 1.0

    # 5. Project Website (Max 0.5)
    if criteria_inputs.get("has_project_website", False):
        score += 0.5

    # 6. Paper Continuation / Lineage (Max 0.5)
    if criteria_inputs.get("is_continuation", False):
        score += 0.5

    # Total max score is 5.5. Normalize between 0 and 5
    # Math: (score / 5.5) * 5.0
    normalized_score = (score / 5.5) * 5.0

    # Round to 2 decimal places
    return round(normalized_score, 2)
