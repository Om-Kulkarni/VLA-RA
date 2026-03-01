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
            list: A list of dictionaries containing paper metadata (title, abstract, url, etc.).
        """
        client = arxiv.Client()
        search_obj = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in client.results(search_obj):
            results.append({
                "title": result.title,
                "abstract": result.summary,
                "url": result.pdf_url or result.entry_id,
                "authors": [author.name for author in result.authors],
                "published": result.published.isoformat() if result.published else None
            })
            
        return results
