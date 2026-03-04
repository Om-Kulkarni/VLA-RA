import pytest
from unittest.mock import patch, MagicMock
from agents.librarian import librarian_node, extract_arxiv_id, filter_relevance


def test_extract_arxiv_id():
    assert extract_arxiv_id("http://arxiv.org/abs/2310.12345v2") == "2310.12345v2"
    assert extract_arxiv_id("https://arxiv.org/pdf/2310.12345") == "2310.12345"


def test_filter_relevance():
    # True because keywords exist and year >= 2023
    assert (
        filter_relevance(
            "VLA for Robotics", "Imitation learning...", "2024-01-01T00:00:00Z"
        )
        is True
    )
    # False because year < 2023
    assert (
        filter_relevance(
            "VLA for Robotics", "Imitation learning...", "2021-01-01T00:00:00Z"
        )
        is False
    )
    # False because no keywords
    assert (
        filter_relevance("Random Paper", "Cooking recipes", "2024-01-01T00:00:00Z")
        is False
    )


@patch("agents.librarian.DatabaseClient")
@patch("os.environ.get")
def test_librarian_node_execution(mock_env_get, mock_db_class):
    mock_env_get.side_effect = lambda k: (
        "dummy_val" if k in ["SUPABASE_URL", "SUPABASE_KEY"] else None
    )

    # Mock database
    mock_db_instance = mock_db_class.return_value
    # Say the first paper is new, second is a duplicate
    mock_db_instance.check_exists.side_effect = [False, True]

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
