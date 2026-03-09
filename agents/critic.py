"""
Critic Node

This module acts as the scoring node, implementing logic based on a weighted rubric.
It delegates deterministic math to core/scoring_logic.py.
"""

import os
import re
import requests
import logging
from typing import Any, Dict
from google import genai
from core.scoring_logic import calculate_rubric_score
from core.database import DatabaseClient
from core.config import get_config

logger = logging.getLogger(__name__)

# Constants for Lookup
ELITE_LABS = [
    "DeepMind",
    "Physical Intelligence",
    "OpenAI",
    "Meta AI",
    "Boston Dynamics",
    "Toyota Research",
    "Berkeley",
    "Stanford",
    "MIT",
    "CMU",
]
CORE_CONFERENCES = ["CoRL", "RSS", "ICRA", "IROS", "CVPR", "NeurIPS", "ICLR", "ICML"]


def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates analyzed components against a strict criteria to generate final scores and rankings.

    Input State:
        - state (Dict[str, Any]): Graph state containing 'analysis_outputs' from the Analyst node.

    Output State:
        - Dict[str, Any]: State subset containing final 'rubric_scores' determining if a piece of research is a "Must-Read".
    """
    logger.info("Critic Node: Starting evaluation of papers.")

    # 1. Setup DB and LLM clients
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and Key must be set in environment variables.")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY must be set in environment variables.")

    db_client = DatabaseClient(url=supabase_url, key=supabase_key)
    ai_client = genai.Client(api_key=gemini_key)

    app_config = get_config()

    # 2. Get Research Manifesto from State
    personal_info = state.get("manifesto", "")
    if not personal_info:
        logger.warning(
            f"Manifesto missing from state, personal relevance might be impaired."
        )

    # Default to filtered_results, fallback to scout_results
    papers_to_evaluate = state.get("filtered_results", [])
    if not papers_to_evaluate:
        papers_to_evaluate = state.get("scout_results", [])

    scores = state.get("rubric_scores", {})

    for paper in papers_to_evaluate:
        title = paper.get("title", "")
        arxiv_id = paper.get("arxiv_id", "")
        # ArXiv returns missing ID as empty or if scout doesn't find it
        if not arxiv_id:
            logger.warning(f"Paper '{title}' has no ArXiv ID. Skipping scoring.")
            continue

        authors = paper.get("authors", [])
        summary = paper.get("summary", "")
        comment = paper.get("comment", "")

        logger.info(f"Evaluating paper: {title} ({arxiv_id})")

        criteria = {}

        # 1. Citations
        try:
            clean_title = title.replace("\n", " ").strip()
            res = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/search",
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

        # 2. Lab & Conf
        author_text = " ".join(authors).lower()
        comment_lower = comment.lower() if comment else ""
        abstract_lower = summary.lower() if summary else ""
        title_lower = title.lower()

        has_elite = any(
            lab.lower() in author_text
            or lab.lower() in comment_lower
            or lab.lower() in abstract_lower
            for lab in ELITE_LABS
        )
        has_conf = any(conf.lower() in comment_lower for conf in CORE_CONFERENCES)

        criteria["has_elite_lab"] = has_elite
        criteria["has_core_conference"] = has_conf

        # 3-6. Structured LLM Extractions: Novelty/Relevance, Links, Website, Lineage
        prompt = f"""
        You are a robotics paper critic. 
        Evaluate the following paper abstract for relevance to this personal research:
        {personal_info}
        
        Paper Title: {title}
        Abstract: {summary}
        Additional Comments: {comment}
        
        Extract the following information and return ONLY a valid JSON object:
        - "relevance_score": A float between 0.0 and 5.0 indicating relevance to personal research (5.0 being extremely highly relevant).
        - "has_code_links": Boolean true if GitHub or HuggingFace links are mentioned in the abstract or comments.
        - "has_project_website": Boolean true if a project website (like .github.io or "project page") is mentioned in the abstract or comments.
        - "is_continuation": Boolean true if the title or abstract indicates this builds upon prior work (e.g., "extension of", "successor to", "v2", "version 2").
        """

        try:
            from google.genai import types
            import json

            response = ai_client.models.generate_content(
                model=app_config.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )

            # The prompt requested JSON output, and we forced the mime_type.
            data_str = response.text.strip()
            # If the model wraps it in ```json ... ```, strip it
            if data_str.startswith("```json"):
                data_str = data_str[7:]
            if data_str.endswith("```"):
                data_str = data_str[:-3]

            llm_result = json.loads(data_str)

            criteria["relevance_score"] = float(llm_result.get("relevance_score", 0.0))
            criteria["has_code_links"] = bool(llm_result.get("has_code_links", False))
            criteria["has_project_website"] = bool(
                llm_result.get("has_project_website", False)
            )
            criteria["is_continuation"] = bool(llm_result.get("is_continuation", False))

        except Exception as e:
            # According to error_handling policy in .agrules, "Do not provide a 'default score' if the LLM fails. Halt and log the error."
            logger.error(f"Failed LLM structured evaluation for paper {title}: {e}")
            raise  # Fail Loudly

        # Calculate Final Score
        final_score = calculate_rubric_score(criteria)
        scores[arxiv_id] = final_score

        logger.info(
            f"Scoring breakdown for {arxiv_id}:\n"
            f"  - Citations: {criteria.get('citations', 0)}\n"
            f"  - Elite Lab: {criteria.get('has_elite_lab', False)}\n"
            f"  - Core Conference: {criteria.get('has_core_conference', False)}\n"
            f"  - Relevance Score: {criteria.get('relevance_score', 0.0)}\n"
            f"  - Code Links: {criteria.get('has_code_links', False)}\n"
            f"  - Project Website: {criteria.get('has_project_website', False)}\n"
            f"  - Continuation: {criteria.get('is_continuation', False)}\n"
            f"  -> Final Score: {final_score}/5.0"
        )

        # Save to database
        db_payload = {
            "arxiv_id": arxiv_id,
            "title": title,
            "priority_score": final_score,
            "metadata": criteria,
        }

        # Update/Upsert the database record. Using standard methods.
        try:
            logger.info(f"Writing score for {arxiv_id} to DB...")
            db_client.update_paper_score(arxiv_id, final_score, criteria)
            logger.info(
                f"Paper {arxiv_id} priority score ({final_score}) successfully saved."
            )
        except Exception as db_err:
            logger.error(f"Failed to update database for {arxiv_id}: {db_err}")
            raise  # DB errors must fail loudly

    # Update State
    return {"rubric_scores": scores}
