import pytest
from unittest.mock import patch, MagicMock
from langchain_core.runnables import RunnableConfig
from agents.scout import scout_node
from core.exceptions import LLMGenerationError


def test_scout_no_query():
    """Test scout with an empty query returns empty results immediately."""
    result = scout_node({"search_query": ""}, config=RunnableConfig())
    assert result == {"scout_results": []}


@patch("agents.scout.genai.Client")
@patch("agents.scout.ArxivTool")
def test_scout_node_execution(mock_arxiv_class, mock_genai_client):
    """
    Test scout node logic. Mocks out LLM and API to verify node behavior without using real tokens.
    """
    # Mock LLM Response
    mock_llm_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    mock_response.text = (
        '{"queries": ["robotics vla imitation", "foundation models bimanual"]}'
    )
    mock_llm_instance.models.generate_content.return_value = mock_response

    # Mock ArXiv Response
    mock_arxiv_instance = mock_arxiv_class.return_value
    mock_arxiv_instance.run.return_value = [
        {
            "url": "http://arxiv.org/abs/1234.5678",
            "title": "Test Paper 1",
            "summary": "Abstract",
        }
    ]

    state = {"search_query": "bimanual manipulation"}
    config = RunnableConfig()

    result = scout_node(state, config)

    assert "scout_results" in result
    results = result["scout_results"]
    assert len(results) > 0
    # Should contain deduplicated papers from arxiv based on URL
    assert results[0]["title"] == "Test Paper 1"
    assert "source_query" in results[0]
