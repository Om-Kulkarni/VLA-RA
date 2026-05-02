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
    Derives all search intent from the research manifesto. If an optional
    'search_query' override is present in state it is used to focus the queries
    further, but the manifesto is the single source of research direction.

    Input State:
        - state (Dict[str, Any]): The current graph state. 'manifesto' is required;
          'search_query' is an optional narrowing hint.

    Output State:
        - Dict[str, Any]: State subset with updated 'scout_results' containing discovered URLs or paper metadata.
    """
    manifesto = state.get("manifesto", "")
    if not manifesto:
        logger.error("Manifesto is empty — cannot generate search queries without research direction.")
        raise ValueError("Manifesto must be present in state. Edit core/manifesto.md to define research goals.")

    # search_query is an optional narrowing hint; the manifesto is the authoritative source.
    search_query = state.get("search_query", "").strip()
    focus_line = (
        f"Additionally narrow the queries toward this specific focus: {search_query}\n"
        if search_query
        else ""
    )

    app_config = get_config()
    client = _make_llm_client(app_config)

    prompt = (
        "You are an expert research strategist. "
        "Based solely on the Research Manifesto below, generate 5 highly specific ArXiv search queries "
        "that will surface the most relevant cutting-edge papers for the defined research goals.\n\n"
        f"Research Manifesto:\n{manifesto}\n\n"
        f"{focus_line}"
        'Return a JSON object with a single key "queries" containing a list of 5 string queries.'
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
