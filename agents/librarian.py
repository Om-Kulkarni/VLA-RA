"""
Librarian Node

This module is responsible for filtering, deduplication, and local database checks.
"""

import os
import re
import json
import logging
from typing import Any, Dict

from openai import OpenAI

from core.database import DatabaseClient
from core.config import get_config

logger = logging.getLogger(__name__)


def extract_arxiv_id(url: str) -> str:
    """
    Extracts the ArXiv ID from a standard ArXiv URL.
    """
    match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)(\d+\.\d+(?:v\d+)?)", url)
    if match:
        return match.group(1)
    return url.split("/")[-1]


def filter_relevance(
    title: str, abstract: str, published: str, manifesto: str, app_config: Any
) -> bool:
    """
    Determines if a paper is relevant by checking alignment with the Central Research
    Manifesto and ensuring it was published in or after 2023.
    Uses DeepSeek V3 via OpenRouter for the alignment check.
    """
    # Recency gate (2023 or later)
    try:
        if published and len(published) >= 4:
            year = int(published[:4])
            if year < 2023:
                return False
        else:
            return False
    except (ValueError, TypeError):
        return False

    # LLM alignment check
    prompt = (
        'You are a strict research Librarian. Your job is to perform a binary check: '
        '"Does this abstract align with our core goals?"\n\n'
        f"Core Goals (Manifesto):\n{manifesto}\n\n"
        f"Paper Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        'Return ONLY a JSON object with a single boolean key "is_relevant". '
        "True if it strongly aligns, False otherwise."
    )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("No OPENROUTER_API_KEY for Librarian, skipping relevance check (defaulting to False).")
        return False

    try:
        client = OpenAI(api_key=api_key, base_url=app_config.openrouter_base_url)
        response = client.chat.completions.create(
            model=app_config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return bool(data.get("is_relevant", False))
    except Exception as e:
        logger.error(f"Failed to use LLM for filter_relevance: {e}")
        return False


def librarian_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters new discoveries and checks against the local database to avoid duplicate processing.

    Input State:
        - state (Dict[str, Any]): The graph state containing raw 'scout_results'.

    Output State:
        - Dict[str, Any]: State subset with 'filtered_results' containing only novel papers to process.
    """
    manifesto = state.get("manifesto", "")
    scout_results = state.get("scout_results", [])

    app_config = get_config()
    db_client = DatabaseClient(db_path=app_config.db_path)

    filtered_results = []

    # --- CRASH RECOVERY: Fetch pending papers ---
    try:
        logger.info("Checking database for stranded 'pending_review' papers...")
        stranded_papers = db_client.get_papers_by_status("pending_review")
        if stranded_papers:
            logger.info(f"Recovered {len(stranded_papers)} 'pending_review' papers from database.")
            for paper in stranded_papers:
                formatted_paper = {
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "title": paper.get("title", ""),
                    "summary": paper.get("summary", ""),
                    "url": paper.get("pdf_url", ""),
                    "authors": [],
                    "comment": "",
                }
                filtered_results.append(formatted_paper)
    except Exception as e:
        logger.error(f"Failed to fetch stranded papers from database: {e}")
        raise  # Fail loudly per .agrules

    if not scout_results:
        return {"filtered_results": filtered_results}

    spaces_left = app_config.max_papers_per_run - len(filtered_results)
    if spaces_left <= 0:
        logger.warning(
            f"Batch limit reached ({app_config.max_papers_per_run}) with stranded papers alone. "
            "Skipping ArXiv processing this run."
        )
        return {"filtered_results": filtered_results[: app_config.max_papers_per_run]}

    for paper in scout_results:
        url = paper.get("url", "")
        arxiv_id = extract_arxiv_id(url)
        paper["arxiv_id"] = arxiv_id

        try:
            logger.info(f"Checking if {arxiv_id} already exists in database...")
            is_dup = db_client.check_exists(arxiv_id)
        except Exception as e:
            logger.error(f"Database check failed for {arxiv_id}: {e}")
            raise  # Fail loudly per .agrules

        if is_dup:
            logger.info(f"Paper {arxiv_id} is a duplicate. Skipping.")
            continue

        title = paper.get("title", "")
        abstract = paper.get("summary", "")
        if "abstract" in paper:
            abstract = paper["abstract"]
        published = paper.get("published", "Unknown")

        if spaces_left <= 0:
            logger.info("Batch limit reached. Stopping new discovery additions.")
            break

        is_relevant = filter_relevance(title, abstract, published, manifesto, app_config)
        if is_relevant:
            db_record = {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": abstract,
                "pdf_url": url,
                "status": "pending_review",
            }
            try:
                db_client.insert_record(db_record)
                logger.info(f"Inserted paper {arxiv_id} into database with status 'pending_review'.")
                filtered_results.append(paper)
                spaces_left -= 1
            except Exception as e:
                logger.error(f"Failed to insert paper {arxiv_id} into database: {e}")
                raise  # Fail loudly per .agrules
        else:
            logger.info(f"Paper {arxiv_id} rejected by relevance/recency check.")

    return {"filtered_results": filtered_results}
