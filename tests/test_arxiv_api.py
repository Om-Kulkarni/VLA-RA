import pytest
from tools.arxiv_api import ArxivTool


def test_arxiv_connection():
    """
    Test the ArXiv API tool connection.
    Searches for a simple known topic.
    """
    tool = ArxivTool()

    results = tool.run(query="vision language action", max_results=1)

    # It should return a list
    assert isinstance(results, list)

    # We expect at least one result for such a broad topic
    assert len(results) > 0

    # Check standard format
    first_result = results[0]
    assert "title" in first_result
    assert "abstract" in first_result
    assert "url" in first_result
    assert "authors" in first_result
