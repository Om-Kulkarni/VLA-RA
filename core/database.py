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
        response = self.supabase.table('research_papers').select('id').eq('arxiv_id', external_id).execute()
        return len(response.data) > 0
    
    def insert_record(self, data: dict):
        """
        Inserts a new processed record into the database. Fails loudly on error.
        
        Args:
            data (dict): The final scored payload to insert.
        """
        # Fails loudly on any Supabase error during insertion
        self.supabase.table('research_papers').insert(data).execute()

    def update_status(self, external_id: str, new_status: str):
        """
        Updates the status of a specific paper. Fails loudly on error.

        Args:
            external_id (str): The ArXiv ID.
            new_status (str): The new status to set ('discovered', 'analyzed', 'rejected').
        """
        self.supabase.table('research_papers').update({'status': new_status}).eq('arxiv_id', external_id).execute()
