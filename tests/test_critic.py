import pytest
from unittest.mock import patch, MagicMock
from agents.critic import critic_node


@patch("agents.critic.genai.Client")
@patch("agents.critic.DatabaseClient")
@patch("agents.critic.requests.get")
@patch("os.environ.get")
@patch("os.path.exists")
@patch("builtins.open")
def test_critic_node_execution(
    mock_open,
    mock_exists,
    mock_env_get,
    mock_requests_get,
    mock_db_class,
    mock_genai_client,
):
    """
    Test critic node execution.
    Mocks LLM, DB, and external requests to prevent hitting external APIS/tokens.
    """
    # Env Setup
    mock_env_get.side_effect = lambda k: (
        "dummy_val" if k in ["SUPABASE_URL", "SUPABASE_KEY", "GEMINI_API_KEY"] else None
    )

    # File Setup
    mock_exists.return_value = True
    mock_file = MagicMock()
    mock_file.read.return_value = "Interested in VLA and robotic manipulation."
    mock_open.return_value.__enter__.return_value = mock_file

    # Request setup (Semantic Scholar)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"citationCount": 50}]}
    mock_requests_get.return_value = mock_response

    # LLM Setup
    mock_llm_instance = mock_genai_client.return_value
    mock_llm_response = MagicMock()
    mock_llm_response.text = "4.5"
    mock_llm_instance.models.generate_content.return_value = mock_llm_response

    input_state = {
        "filtered_results": [
            {
                "title": "Amazing Bimanual VLA",
                "arxiv_id": "2401.99999",
                "authors": ["John Doe", "Jane Smith from DeepMind"],
                "summary": "We built a new foundation model for bimanual robots evaluated on CoRL benchmarks. See project page at example.github.io",
                "comment": "Accepted at CoRL 2024",
            }
        ]
    }

    result = critic_node(input_state)

    # Assertions
    assert "rubric_scores" in result
    scores = result["rubric_scores"]
    assert "2401.99999" in scores
    assert isinstance(scores["2401.99999"], float)

    # Verify DB was called to save
    mock_db_instance = mock_db_class.return_value
    mock_db_instance.update_paper_score.assert_called_once()
