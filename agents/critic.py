"""
Critic Node

This module acts as the scoring node, implementing logic based on a weighted rubric.
It delegates deterministic math to core/scoring_logic.py.

Labs and conferences used for heuristic matching are parsed from the research manifesto
at runtime — they must NOT be hardcoded here. Edit core/manifesto.md to change them.
"""

import os
import json
import logging
import requests
from typing import Any, Dict

from openai import OpenAI

from core.scoring_logic import calculate_rubric_score
from core.database import DatabaseClient
from core.config import get_config
from core.manifesto_parser import parse_priority_labs, parse_target_conferences

logger = logging.getLogger(__name__)


def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates filtered papers against a strict weighted rubric to generate final scores.

    Labs and conferences for heuristic matching are read from the manifesto (GraphState),
    never from hardcoded constants — enabling zero-code research domain switches.

    Input State:
        - state (Dict[str, Any]): Graph state containing 'filtered_results' from the
          Librarian node and 'manifesto' from core/manifesto.md.

    Output State:
        - Dict[str, Any]: State subset containing 'rubric_scores' — a dict of
          {arxiv_id: float} priority scores normalised to 0–5.
    """
    logger.info("Critic Node: Starting evaluation of papers.")

    # 1. Setup DB and LLM clients
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY must be set in environment variables.")

    app_config = get_config()
    db_client = DatabaseClient(db_path=app_config.db_path)
    ai_client = OpenAI(api_key=api_key, base_url=app_config.openrouter_base_url)

    # 2. Get Research Manifesto from State
    manifesto = state.get("manifesto", "")
    if not manifesto:
        logger.warning("Manifesto missing from state. Lab/conference heuristics will match nothing.")

    # 3. Parse labs and conferences from manifesto — single source of truth
    elite_labs = parse_priority_labs(manifesto)
    core_conferences = parse_target_conferences(manifesto)
    logger.info(f"Loaded {len(elite_labs)} priority labs and {len(core_conferences)} target conferences from manifesto.")

    # 4. Resolve papers to evaluate
    papers_to_evaluate = state.get("filtered_results", [])
    if not papers_to_evaluate:
        papers_to_evaluate = state.get("scout_results", [])

    scores = state.get("rubric_scores", {})

    for paper in papers_to_evaluate:
        title = paper.get("title", "")
        arxiv_id = paper.get("arxiv_id", "")
        if not arxiv_id:
            logger.warning(f"Paper '{title}' has no ArXiv ID. Skipping scoring.")
            continue

        authors = paper.get("authors", [])
        summary = paper.get("summary", "")
        comment = paper.get("comment", "")

        logger.info(f"Evaluating paper: {title} ({arxiv_id})")

        criteria = {}

        # ---------------------------------------------------------------
        # Criterion 1: Citations (Semantic Scholar)
        # ---------------------------------------------------------------
        try:
            clean_title = title.replace("\n", " ").strip()
            res = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": clean_title, "fields": "citationCount", "limit": 1},
                timeout=10,
            )
            if res.status_code == 200:
                data = res.json()
                if data.get("data") and len(data["data"]) > 0:
                    criteria["citations"] = data["data"][0].get("citationCount", 0)
                else:
                    criteria["citations"] = 0
            else:
                criteria["citations"] = 0
        except Exception as e:
            logger.warning(f"Semantic Scholar lookup failed for {arxiv_id}: {e}")
            criteria["citations"] = 0

        # ---------------------------------------------------------------
        # Criterion 2: Lab & Conference heuristics (manifesto-driven)
        # ---------------------------------------------------------------
        author_text = " ".join(authors).lower()
        comment_lower = comment.lower() if comment else ""
        abstract_lower = summary.lower() if summary else ""

        criteria["has_elite_lab"] = any(
            lab.lower() in author_text
            or lab.lower() in comment_lower
            or lab.lower() in abstract_lower
            for lab in elite_labs
        )
        criteria["has_core_conference"] = any(
            conf.lower() in comment_lower for conf in core_conferences
        )

        # ---------------------------------------------------------------
        # Criteria 3–7: Structured LLM extraction
        # ---------------------------------------------------------------
        prompt = (
            "You are an expert research evaluator.\n"
            "Evaluate the paper below against this Research Manifesto and return ONLY a valid JSON object.\n\n"
            f"Research Manifesto:\n{manifesto}\n\n"
            f"Paper Title: {title}\n"
            f"Abstract: {summary}\n"
            f"ArXiv Comments: {comment}\n\n"
            "Extract the following fields. Be precise — scores drive automated ranking.\n\n"
            '- "task_relevance": float 0.0–5.0. How strongly does this paper address tasks '
            "defined in the manifesto's Core Interests? (5 = perfect match, 0 = entirely unrelated)\n"
            '- "method_novelty": float 0.0–5.0. Does this introduce a genuinely new technique, '
            "architecture, or insight? (5 = paradigm-shifting, 0 = no novel contribution)\n"
            '- "embodiment_match": float 0.0–5.0. Is the target hardware, robot morphology, or '
            "action space relevant to the manifesto's embodiment interests? "
            "(5 = exact match, 0 = completely different domain)\n"
            '- "novelty_score": float 0.0–5.0. Overall assessment of how frontier-pushing this '
            "work is, independent of relevance. (5 = major community milestone, 0 = incremental ablation)\n"
            '- "code_maturity": integer 0, 1, or 2.\n'
            "  0 = No code release mentioned.\n"
            "  1 = GitHub repository link present (code only, no weights or demo).\n"
            "  2 = Model weights, HuggingFace release, or live interactive demo available.\n"
            '- "has_project_website": boolean. True if a dedicated project page '
            "(e.g., a .github.io site or 'project page' link) is mentioned.\n"
            '- "is_continuation": boolean. True if the paper explicitly builds upon or extends '
            'prior named work (e.g., "v2", "extension of X", "successor to Y"). '
            "Metadata only — does not affect the score."
        )

        try:
            response = ai_client.chat.completions.create(
                model=app_config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            llm_result = json.loads(response.choices[0].message.content)

            criteria["task_relevance"] = float(llm_result.get("task_relevance", 0.0))
            criteria["method_novelty"] = float(llm_result.get("method_novelty", 0.0))
            criteria["embodiment_match"] = float(llm_result.get("embodiment_match", 0.0))
            criteria["novelty_score"] = float(llm_result.get("novelty_score", 0.0))
            criteria["code_maturity"] = int(llm_result.get("code_maturity", 0))
            criteria["has_project_website"] = bool(llm_result.get("has_project_website", False))
            # Metadata only — passed to DB but excluded from scoring
            criteria["is_continuation"] = bool(llm_result.get("is_continuation", False))

        except Exception as e:
            # Per .agrules: "Do not provide a 'default score' if the LLM fails. Halt and log."
            logger.error(f"Failed LLM structured evaluation for paper {title}: {e}")
            raise  # Fail Loudly

        # ---------------------------------------------------------------
        # Calculate Final Score (deterministic math only — no LLM here)
        # ---------------------------------------------------------------
        final_score = calculate_rubric_score(criteria)
        scores[arxiv_id] = final_score

        composite_relevance = (
            criteria.get("task_relevance", 0.0)
            + criteria.get("method_novelty", 0.0)
            + criteria.get("embodiment_match", 0.0)
        ) / 3.0

        logger.info(
            f"Scoring breakdown for {arxiv_id}:\n"
            f"  - Citations:            {criteria.get('citations', 0)}\n"
            f"  - Elite Lab:            {criteria.get('has_elite_lab', False)}\n"
            f"  - Core Conference:      {criteria.get('has_core_conference', False)}\n"
            f"  - Task Relevance:       {criteria.get('task_relevance', 0.0):.1f}/5\n"
            f"  - Method Novelty:       {criteria.get('method_novelty', 0.0):.1f}/5\n"
            f"  - Embodiment Match:     {criteria.get('embodiment_match', 0.0):.1f}/5\n"
            f"  - Composite Relevance:  {composite_relevance:.2f}/5\n"
            f"  - Novelty Score:        {criteria.get('novelty_score', 0.0):.1f}/5\n"
            f"  - Code Maturity:        {criteria.get('code_maturity', 0)}/2\n"
            f"  - Project Website:      {criteria.get('has_project_website', False)}\n"
            f"  - Continuation (meta):  {criteria.get('is_continuation', False)}\n"
            f"  -> Final Score:         {final_score}/5.0"
        )

        # ---------------------------------------------------------------
        # Persist to local database
        # ---------------------------------------------------------------
        try:
            logger.info(f"Writing score for {arxiv_id} to DB...")
            db_client.update_paper_score(arxiv_id, final_score, criteria)
            logger.info(f"Paper {arxiv_id} priority score ({final_score}) successfully saved.")
        except Exception as db_err:
            logger.error(f"Failed to update database for {arxiv_id}: {db_err}")
            raise  # DB errors must fail loudly per .agrules

    return {"rubric_scores": scores}
