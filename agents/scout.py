"""
Scout Node

This module acts as the dynamic search architect for the LangGraph workflow.
It performs LLM-driven queries via OpenRouter (DeepSeek V3) to find relevant
research papers and repositories.
"""

import os
import json
import logging
from typing import Any, Dict

from openai import OpenAI
from langchain_core.runnables import RunnableConfig

from tools.arxiv_api import ArxivTool
from core.exceptions import LLMGenerationError, ExternalAPIError
from core.config import get_config

logger = logging.getLogger(__name__)


def _make_llm_client(config: Any) -> OpenAI:
    """Constructs an OpenAI-compatible client pointed at OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY must be set in environment variables.")
    return OpenAI(api_key=api_key, base_url=config.openrouter_base_url)


def scout_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes the Scout logic to query external sources.

    Input State:
        - state (Dict[str, Any]): The current graph state containing search intents or keywords.

    Output State:
        - Dict[str, Any]: State subset with updated 'scout_results' containing discovered URLs or paper metadata.
    """
    search_query = state.get("search_query", "")
    manifesto = state.get("manifesto", "")
    if not search_query:
        return {"scout_results": []}

    app_config = get_config()
    client = _make_llm_client(app_config)

    prompt = (
        f"You are a Lead Robotics Researcher. Expand the following topic into 5 highly "
        f"specific search queries for ArXiv: {search_query}.\n"
        f"Align these queries with this core Manifesto:\n{manifesto}\n\n"
        f'Return a JSON object with a single key "queries" containing a list of 5 string queries.'
    )

    try:
        response = client.chat.completions.create(
            model=app_config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        queries = data.get("queries", [])
    except Exception as e:
        logger.error(f"Failed to generate queries from LLM: {e}")
        raise LLMGenerationError(f"LLM intent generation failed: {e}") from e

    arxiv_tool = ArxivTool()
    all_results = []
    seen_urls = set()

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
