import pytest
from unittest.mock import patch, MagicMock
from agents.librarian import librarian_node, extract_arxiv_id, filter_relevance


def test_extract_arxiv_id():
    assert extract_arxiv_id("http://arxiv.org/abs/2310.12345v2") == "2310.12345v2"
    assert extract_arxiv_id("https://arxiv.org/pdf/2310.12345") == "2310.12345"


@patch("google.genai.Client")
@patch("os.environ.get")
def test_filter_relevance(mock_env_get, mock_genai_client):
    mock_env_get.side_effect = lambda k: "dummy_val" if k == "GEMINI_API_KEY" else None

    # Mock LLM response to simulate alignment
    mock_llm_instance = mock_genai_client.return_value
    mock_llm_response = MagicMock()
    mock_llm_response.text = '{"is_relevant": true}'
    mock_llm_instance.models.generate_content.return_value = mock_llm_response

    # True based on mock and year >= 2023
    assert (
        filter_relevance(
            "VLA for Robotics",
            "Imitation learning...",
            "2024-01-01T00:00:00Z",
            "dummy manifesto",
        )
        is True
    )

    # False because year < 2023 (caught before LLM)
    assert (
        filter_relevance(
            "VLA for Robotics",
            "Imitation learning...",
            "2021-01-01T00:00:00Z",
            "dummy manifesto",
        )
        is False
    )

    # False based on LLM saying false
    mock_llm_response.text = '{"is_relevant": false}'
    assert (
        filter_relevance(
            "Random Paper", "Cooking recipes", "2024-01-01T00:00:00Z", "dummy manifesto"
        )
        is False
    )


@patch("agents.librarian.filter_relevance")
@patch("agents.librarian.DatabaseClient")
@patch("os.environ.get")
def test_librarian_node_execution(mock_env_get, mock_db_class, mock_filter):
    mock_env_get.side_effect = lambda k: (
        "dummy_val" if k in ["SUPABASE_URL", "SUPABASE_KEY"] else None
    )

    # Mock database
    mock_db_instance = mock_db_class.return_value
    mock_db_instance.check_exists.side_effect = [False, True]
    mock_db_instance.get_papers_by_status.return_value = []

    # Mock filter
    mock_filter.return_value = True

    input_state = {
        "scout_results": [
            {
                "url": "http://arxiv.org/abs/2401.00001",
                "title": "A Great VLA Paper",
                "summary": "This is a bimanual manipulation paper.",
                "published": "2024-01-01T10:00:00Z",
            },
            {
                "url": "http://arxiv.org/abs/2401.00002",
                "title": "Duplicate VLA Paper",
                "summary": "This is a bimanual manipulation paper.",
                "published": "2024-01-02T10:00:00Z",
            },
        ]
    }

    result = librarian_node(input_state)
    filtered = result["filtered_results"]

    assert len(filtered) == 1
    assert filtered[0]["arxiv_id"] == "2401.00001"
    mock_db_instance.insert_record.assert_called_once()
