"""
Database Management Interface

This module interacts with the Supabase/Postgres backend for reading and writing processed data.
Any database failure must trigger an explicit crash/halt, not caught silently.
"""

from supabase import create_client, Client, ClientOptions
import os


class DatabaseClient:
    """
    Client wrapper for interacting with Supabase securely.
    """

    def __init__(self, url: str, key: str):
        """
        Initializes the Supabase client connection.

        Args:
            url (str): The Supabase project URL.
            key (str): The Supabase API/Service key.
        """
        # Validates that url and key are provided (Fail loudly)
        if not url or not key:
            raise ValueError("Supabase URL and Key must be provided.")

        # Adding a 10s timeout enforces Fail Loudly if DB host is unreachable
        options = ClientOptions(postgrest_client_timeout=10)
        self.supabase: Client = create_client(url, key, options=options)

    def check_exists(self, external_id: str) -> bool:
        """
        Checks if a paper already exists in the Postgres database.

        Args:
            external_id (str): The unique identifier (e.g., ArXiv ID) to check.

        Returns:
            bool: True if it exists, False otherwise.
        """
        # If this fails, it natively raises an exception (e.g., mapped by Supabase Python client),
        # fulfilling the "fail loudly" policy.
        response = (
            self.supabase.table("research_papers")
            .select("id")
            .eq("arxiv_id", external_id)
            .execute()
        )
        return len(response.data) > 0

    def insert_record(self, data: dict):
        """
        Inserts a new processed record into the database. Fails loudly on error.

        Args:
            data (dict): The final scored payload to insert.
        """
        # Fails loudly on any Supabase error during insertion
        self.supabase.table("research_papers").insert(data).execute()

    def get_papers_by_status(self, status: str) -> list[dict]:
        """
        Fetches all papers from the database that match the given status.

        Args:
            status (str): The status to filter by (e.g., 'pending_review').

        Returns:
            list[dict]: A list of paper records matching the status.
        """
        response = (
            self.supabase.table("research_papers")
            .select("*")
            .eq("status", status)
            .execute()
        )
        return response.data

    def update_status(self, external_id: str, new_status: str):
        """
        Updates the status of a specific paper. Fails loudly on error.

        Args:
            external_id (str): The ArXiv ID.
            new_status (str): The new status to set ('pending_review', 'scored', 'rejected').
        """
        self.supabase.table("research_papers").update({"status": new_status}).eq(
            "arxiv_id", external_id
        ).execute()

    def update_paper_score(self, external_id: str, score: float, metadata: dict):
        """
        Updates the score and metadata of a specific paper. Fails loudly on error.

        Args:
            external_id (str): The ArXiv ID.
            score (float): The final priority score.
            metadata (dict): The dict of extracted heuristic values.
        """
        update_data = {"score": score, "status": "scored", "metadata": metadata}

        # Check if the extracted data explicitly includes links
        if "github_link" in metadata:
            update_data["github_link"] = metadata.get("github_link")

        if "project_website" in metadata:
            update_data["project_website"] = metadata.get("project_website")

        self.supabase.table("research_papers").update(update_data).eq(
            "arxiv_id", external_id
        ).execute()
