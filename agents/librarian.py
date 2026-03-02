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

logger = logging.getLogger(__name__)

def extract_arxiv_id(url: str) -> str:
    """
    Extracts the ArXiv ID from a standard ArXiv URL.
    """
    match = re.search(r'(?:arxiv\.org/(?:abs|pdf)/|arxiv:)(\d+\.\d+(?:v\d+)?)', url)
    if match:
        return match.group(1)
    # Fallback to the last part of the URL (entry_id fallback)
    return url.split('/')[-1]

KEYWORDS = [
    "vision-language-action", "vla", "imitation learning", 
    "foundation model", "robotics", "manipulation", "bimanual"
]

def filter_relevance(title: str, abstract: str, published: str) -> bool:
    """
    Determines if a paper is relevant by checking for specific keywords 
    in the title or abstract, and ensuring it was published in or after 2023.
    """
    # Check recency (2023 or later)
    # The 'published' string from arxiv is typically in ISO format like '2023-11-20T18:32:00Z'
    try:
        if published and len(published) >= 4:
            year = int(published[:4])
            if year < 2023:
                return False
        else:
            return False
    except (ValueError, TypeError):
        return False
        
    # Check keywords
    text_to_search = f"{title} {abstract}".lower()
    for kw in KEYWORDS:
        if kw in text_to_search:
            return True
            
    return False

def librarian_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters new discoveries and checks against the Supabase database to avoid duplicate processing.
    
    Input State:
        - state (Dict[str, Any]): The graph state containing raw 'scout_results'.
        
    Output State:
        - Dict[str, Any]: State subset with 'filtered_results' containing only novel papers/repos to process.
    """
    scout_results = state.get("scout_results", [])
    if not scout_results:
        return {"filtered_results": []}

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    db_client = DatabaseClient(url=supabase_url, key=supabase_key)
    filtered_results = []
    
    for paper in scout_results:
        url = paper.get("url", "")
        # Deduplication Check
        arxiv_id = extract_arxiv_id(url)
        paper['arxiv_id'] = arxiv_id  # Enrich paper dict with arxiv_id
        
        try:
            logger.info(f"Checking if {arxiv_id} already exists in database...")
            is_dup = db_client.check_exists(arxiv_id) 
        except Exception as e: 
            logger.error(f"Database check failed for {arxiv_id}: {e}") 
            raise # Fail loudly per .agrules
            
        if is_dup:
            logger.info(f"Paper {arxiv_id} is a duplicate. Skipping.")
            continue
            
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        published = paper.get("published", "Unknown")
        
        is_relevant = filter_relevance(title, abstract, published)
        if is_relevant:
            # Map paper dict to Supabase DB schema
            db_record = {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": abstract,
                "pdf_url": url,
                "status": "discovered"
            }
            try:
                db_client.insert_record(db_record)
                logger.info(f"Inserted paper {arxiv_id} into database with status 'discovered'.")
            except Exception as e:
                logger.error(f"Failed to insert paper {arxiv_id} into database: {e}")
                raise  # Fail loudly per .agrules
                
            filtered_results.append(paper)
        else:
            logger.info(f"Paper {arxiv_id} rejected by relevance/recency check.")
            
    return {"filtered_results": filtered_results}
