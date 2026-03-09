"""
Librarian Node

This module is responsible for filtering, deduplication, and database checks.
"""

import os
import re
import json
import logging
from typing import Any, Dict

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
    # Fallback to the last part of the URL (entry_id fallback)
    return url.split("/")[-1]


def filter_relevance(
    title: str, abstract: str, published: str, manifesto: str, llm_model: str
) -> bool:
    """
    Determines if a paper is relevant by checking for alignment with the Central Research Manifesto
    and ensuring it was published in or after 2023. Uses Gemini for the alignment check.
    """
    # Check recency (2023 or later)
    try:
        if published and len(published) >= 4:
            year = int(published[:4])
            if year < 2023:
                return False
        else:
            return False
    except (ValueError, TypeError):
        return False

    # Check alignment using Gemini
    prompt = f"""You are a strict research Librarian. Your job is to perform a binary check: "Does this abstract align with our core goals?"
    
    Core Goals (Manifesto):
    {manifesto}
    
    Paper Title: {title}
    Abstract: {abstract}
    
    Return ONLY a JSON object with a single boolean key "is_relevant". True if it strongly aligns, False otherwise.
    """

    try:
        from google import genai
        from google.genai import types
        import json
        import os

        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            logger.warning(
                "No Gemini key for Librarian, skipping relevance check (defaulting to False)"
            )
            return False

        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        data_str = response.text.strip()
        if data_str.startswith("```json"):
            data_str = data_str[7:]
        if data_str.endswith("```"):
            data_str = data_str[:-3]

        data = json.loads(data_str)
        return bool(data.get("is_relevant", False))

    except Exception as e:
        logger.error(f"Failed to use LLM for filter_relevance: {e}")
        return False


def librarian_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters new discoveries and checks against the Supabase database to avoid duplicate processing.

    Input State:
        - state (Dict[str, Any]): The graph state containing raw 'scout_results'.

    Output State:
        - Dict[str, Any]: State subset with 'filtered_results' containing only novel papers/repos to process.
    """
    manifesto = state.get("manifesto", "")
    scout_results = state.get("scout_results", [])

    app_config = get_config()
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    db_client = DatabaseClient(url=supabase_url, key=supabase_key)

    filtered_results = []

    # --- CRASH RECOVERY: Fetch pending papers ---
    try:
        logger.info("Checking database for stranded 'pending_review' papers...")
        stranded_papers = db_client.get_papers_by_status("pending_review")
        if stranded_papers:
            logger.info(
                f"Recovered {len(stranded_papers)} 'pending_review' papers from Database."
            )
            for paper in stranded_papers:
                # Format to match ArXiv dict output so Critic can process it seamlessly
                formatted_paper = {
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "title": paper.get("title", ""),
                    "summary": paper.get("summary", ""),
                    "url": paper.get("pdf_url", ""),
                    "authors": [],  # We don't save authors to DB currently
                    "comment": "",
                }
                filtered_results.append(formatted_paper)
    except Exception as e:
        logger.error(f"Failed to fetch stranded papers from Database: {e}")
        raise  # Fail loudly per .agrules

    # Process new discoveries
    if not scout_results:
        return {"filtered_results": filtered_results}

    # Track how many API calls we can still afford for validation + criticism processing
    spaces_left = app_config.max_papers_per_run - len(filtered_results)
    if spaces_left <= 0:
        logger.warning(
            f"Batch limit reached ({app_config.max_papers_per_run}) with stranded database papers alone. Skipping ArXiv processing this run."
        )
        # Slice to ensure we do not exceed the limit
        return {"filtered_results": filtered_results[: app_config.max_papers_per_run]}

    for paper in scout_results:
        url = paper.get("url", "")
        # Deduplication Check
        arxiv_id = extract_arxiv_id(url)
        paper["arxiv_id"] = arxiv_id  # Enrich paper dict with arxiv_id

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
        abstract = paper.get(
            "summary", ""
        )  # ArXiv uses 'summary' not 'abstract' usually, though scout might map it
        if "abstract" in paper:
            abstract = paper["abstract"]
        published = paper.get("published", "Unknown")

        if spaces_left <= 0:
            logger.info("Batch limit reached. Stopping new discovery additions.")
            break

        is_relevant = filter_relevance(
            title, abstract, published, manifesto, app_config.llm_model
        )
        if is_relevant:
            # Map paper dict to Supabase DB schema
            db_record = {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": abstract,
                "pdf_url": url,
                "status": "pending_review",
            }
            try:
                db_client.insert_record(db_record)
                logger.info(
                    f"Inserted paper {arxiv_id} into database with status 'pending_review'."
                )
                filtered_results.append(paper)
                spaces_left -= 1
            except Exception as e:
                logger.error(f"Failed to insert paper {arxiv_id} into database: {e}")
                raise  # Fail loudly per .agrules
        else:
            logger.info(f"Paper {arxiv_id} rejected by relevance/recency check.")

    return {"filtered_results": filtered_results}
