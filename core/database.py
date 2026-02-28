"""
Database Management Interface

This module interacts with the Supabase/Postgres backend for reading and writing processed data.
Any database failure must trigger an explicit crash/halt, not caught silently.
"""

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
        pass
    
    def check_exists(self, external_id: str) -> bool:
        """
        Checks if a paper or repository already exists in the Postgres database.
        
        Args:
            external_id (str): The unique identifier (e.g., ArXiv ID) to check.
            
        Returns:
            bool: True if it exists, False otherwise.
        """
        return False
    
    def insert_record(self, data: dict):
        """
        Inserts a new processed record into the database. Fails loudly on error.
        
        Args:
            data (dict): The final scored payload to insert.
        """
        pass
