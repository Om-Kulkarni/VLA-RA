"""
Scoring Logic

This module provides the deterministic math and weighted rubric calculations
for evaluating papers and repositories. It operates on LLM-generated raw inputs.

According to O (Open/Closed principle), new scoring criteria should be added here
without modifying the Agent graph logic.

Rubric (max raw score = 6.0, normalised to 5.0):
  1. Citations                          — max 1.0
  2. Lab Impact & Target Conferences    — max 1.0
  3. Composite Relevance (3 sub-dims)   — max 1.5
  4. Novelty                            — max 1.0
  5. Code Maturity (0/1/2)             — max 1.0
  6. Project Website                    — max 0.5
  is_continuation                       — metadata only, NOT scored
"""

from typing import Dict, Any

_MAX_RAW_SCORE = 6.0


def calculate_rubric_score(criteria_inputs: Dict[str, Any]) -> float:
    """
    Calculates the final deterministic score based on a weighted rubric.

    Args:
        criteria_inputs (Dict[str, Any]): A dictionary of raw scores or categorical
            values provided by the Critic node. Expected keys:

            citations (int):
                Citation count from Semantic Scholar.
            has_elite_lab (bool):
                True if any author affiliation matches a Priority Lab from the manifesto.
            has_core_conference (bool):
                True if the paper comment references a Target Conference from the manifesto.
            task_relevance (float, 0–5):
                LLM score — does the paper address tasks in the core interest areas?
            method_novelty (float, 0–5):
                LLM score — does it introduce a genuinely new technique or architecture?
            embodiment_match (float, 0–5):
                LLM score — is the target hardware/embodiment relevant to our research?
            novelty_score (float, 0–5):
                LLM score — paradigm-shifting contribution vs. incremental improvement.
            code_maturity (int, 0|1|2):
                0 = no code, 1 = GitHub link only, 2 = weights/demo/HuggingFace release.
            has_project_website (bool):
                True if a dedicated project page is mentioned.
            is_continuation (bool):
                Metadata only — whether this extends prior work. NOT included in scoring.

    Returns:
        float: Final weighted score normalised to 0–5, rounded to 2 decimal places.
    """
    score = 0.0

    # 1. Citations (Max 1.0)
    # Thresholds tuned for post-2023 papers which rarely exceed 20 citations quickly.
    citations = int(criteria_inputs.get("citations", 0))
    if citations >= 20:
        score += 1.0
    elif citations >= 5:
        score += 0.5
    elif citations > 0:
        score += 0.2

    # 2. Lab Impact & Target Conferences (Max 1.0)
    # Labs and conferences are parsed from manifesto at Critic runtime — never hardcoded.
    lab_conf_score = 0.0
    if criteria_inputs.get("has_elite_lab", False):
        lab_conf_score += 0.5
    if criteria_inputs.get("has_core_conference", False):
        lab_conf_score += 0.5
    score += lab_conf_score

    # 3. Composite Relevance — 3 LLM sub-dimensions averaged (Max 1.5)
    # Three independent axes reduce single-number hallucination variance.
    sub_scores = [
        criteria_inputs.get("task_relevance", 0.0),
        criteria_inputs.get("method_novelty", 0.0),
        criteria_inputs.get("embodiment_match", 0.0),
    ]
    composite = sum(max(0.0, min(5.0, float(s))) for s in sub_scores) / 3.0
    score += (composite / 5.0) * 1.5

    # 4. Novelty (Max 1.0)
    # Separate from relevance — captures paradigm-shifting vs. incremental contribution.
    novelty = criteria_inputs.get("novelty_score", 0.0)
    score += (max(0.0, min(5.0, float(novelty))) / 5.0) * 1.0

    # 5. Code Maturity (Max 1.0)
    # Graduated: code-only (0.5) vs. weights/demo available (1.0).
    code_maturity = int(criteria_inputs.get("code_maturity", 0))
    score += {0: 0.0, 1: 0.5, 2: 1.0}.get(code_maturity, 0.0)

    # 6. Project Website (Max 0.5)
    if criteria_inputs.get("has_project_website", False):
        score += 0.5

    # NOTE: is_continuation is NOT scored — it is metadata only, stored to DB for
    # analyst context. A sequel to a great paper is great; a sequel to a weak one
    # is not. The signal has no reliable direction, so we exclude it.

    # Normalise: raw max is 6.0, output range is 0–5.
    normalized_score = (score / _MAX_RAW_SCORE) * 5.0
    return round(normalized_score, 2)
