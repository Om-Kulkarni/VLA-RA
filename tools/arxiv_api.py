"""
ArXiv API Wrapper

This module provides a search wrapper to interface with ArXiv and potentially OpenReview.
"""
import arxiv
from core.base_tool import BaseTool


class ArxivTool(BaseTool):
    """
    A tool configured to search for and retrieve metadata from ArXiv.
    """

    def run(self, query: str, max_results: int = 5) -> list:
        """
        Executes a search query against the ArXiv API.

        Args:
            query (str): The search intent or keywords.
            max_results (int, optional): The maximum number of results to return. Defaults to 5.

        Returns:
            list: A list of dicts containing paper metadata (title, summary, url, authors, etc.).
        """
        client = arxiv.Client()
        search_obj = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for result in client.results(search_obj):
            results.append(self._result_to_dict(result))

        return results

    def fetch_by_id(self, arxiv_id: str) -> dict:
        """
        Fetches the full metadata for a single paper by its ArXiv ID.
        Used in express mode to bypass the Scout/Librarian pipeline.

        Args:
            arxiv_id (str): The ArXiv paper ID (e.g. '2601.16163').

        Returns:
            dict: Paper metadata dict in the standard pipeline format.

        Raises:
            ValueError: If no paper is found for the given ID.
        """
        client = arxiv.Client()
        search_obj = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search_obj))
        if not results:
            raise ValueError(f"No paper found on ArXiv for ID: {arxiv_id}")
        return self._result_to_dict(results[0])

    @staticmethod
    def _result_to_dict(result: arxiv.Result) -> dict:
        """Converts an arxiv.Result object to the standard pipeline dict format."""
        # Extract bare ID (strip version suffix like 'v1')
        entry_id = result.entry_id  # e.g. "http://arxiv.org/abs/2601.16163v1"
        bare_id = entry_id.split("/")[-1].split("v")[0]  # "2601.16163"

        return {
            "title": result.title,
            "summary": result.summary,       # pipeline uses 'summary'
            "abstract": result.summary,       # alias for compatibility
            "url": result.pdf_url or result.entry_id,
            "authors": [author.name for author in result.authors],
            "published": result.published.isoformat() if result.published else None,
            "comment": result.comment or "",
            "arxiv_id": bare_id,
        }
