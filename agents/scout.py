"""
Scout Node

This module acts as the dynamic search architect for the LangGraph workflow.
It performs LLM-driven queries to find relevant research papers and repositories.
"""
import json
import logging
from typing import Any, Dict
from google import genai
from google.genai import types

from tools.arxiv_api import ArxivTool
from core.exceptions import LLMGenerationError, ExternalAPIError

logger = logging.getLogger(__name__)

def scout_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the Scout logic to query external sources.
    
    Input State:
        - state (Dict[str, Any]): The current graph state containing search intents or keywords.
        
    Output State:
        - Dict[str, Any]: State subset with updated 'scout_results' containing discovered URLs or paper metadata.
    """
    search_query = state.get("search_query", "")
    if not search_query:
        return {"scout_results": []}
        
    # Generate intent-based queries
    prompt = f"""You are a Lead Robotics Researcher. Expand the following topic into 5 highly specific search queries for ArXiv: {search_query}. Focus on VLA, Foundation Models for Robotics, and Imitation Learning.
Return a JSON object with a single key "queries" containing a list of 5 string queries."""
    
    client = genai.Client()
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        data = json.loads(response.text)
        queries = data.get("queries", [])
    except Exception as e:
        logger.error(f"Failed to generate queries from LLM: {e}")
        raise LLMGenerationError(f"LLM intent generation failed: {e}") from e
        
    arxiv_tool = ArxivTool()
    all_results = []
    seen_urls = set()
    
    # We query ArXiv for each generated intent
    for q in queries:
        try:
            results = arxiv_tool.run(query=q, max_results=3)
            for res in results:
                if res["url"] not in seen_urls:
                    seen_urls.add(res["url"])
                    res["source_query"] = q
                    all_results.append(res)
        except Exception as e:
            logger.error(f"ArXiv search failed for query '{q}': {e}")
            raise ExternalAPIError(f"ArXiv API error: {e}") from e
            
    return {"scout_results": all_results}
