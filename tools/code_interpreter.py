"""
Code Interpreter

This module provides analysis capabilities for GitHub READMEs, Requirements,
and raw code using external APIs or local execution.
"""


class CodeInterpreterTool:
    """
    Tool to inspect and evaluate code repositories.
    """

    def analyze_repository(self, repo_url: str) -> dict:
        """
        Fetches and analyzes a GitHub repository's content (e.g., README.md, requirements.txt).

        Args:
            repo_url (str): The URL of the target GitHub repository.

        Returns:
            dict: An analysis payload containing structural summary and dependencies.
        """
        import requests
        import re
        import logging

        logger = logging.getLogger(__name__)

        # Clean URL and extract owner/repo
        # e.g., https://github.com/owner/repo
        match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            logger.warning(f"Invalid or unsupported GitHub URL: {repo_url}")
            return {"readme": "", "requirements": "", "error": "Invalid GitHub URL"}

        owner = match.group(1)
        repo = match.group(2).replace(".git", "")

        base_raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main"
        master_raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master"

        payload = {"readme": "", "requirements": ""}

        # Fetch README
        for url in [
            f"{base_raw_url}/README.md",
            f"{master_raw_url}/README.md",
            f"{base_raw_url}/readme.md",
        ]:
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    payload["readme"] = res.text
                    break
            except Exception as e:
                logger.debug(f"Failed to fetch {url}: {e}")

        # Fetch Requirements
        for url in [
            f"{base_raw_url}/requirements.txt",
            f"{master_raw_url}/requirements.txt",
        ]:
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    payload["requirements"] = res.text
                    break
            except Exception as e:
                logger.debug(f"Failed to fetch {url}: {e}")

        return payload
