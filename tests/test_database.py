import os
import pytest
from core.database import DatabaseClient
from dotenv import load_dotenv


def test_supabase_connection():
    load_dotenv()
    """
    Test the Supabase database connection.
    This does not create or mutate records, it just verifies the connection works
    by checking for existence.
    """
    # Requires SUPABASE_URL and SUPABASE_KEY in environment
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    assert supabase_url is not None, "SUPABASE_URL not set in environment."
    assert supabase_key is not None, "SUPABASE_KEY not set in environment."

    client = DatabaseClient(url=supabase_url, key=supabase_key)

    # Try checking for a dummy ID.
    # If the database is unreachable, this will fail loudly as desired.
    # It returns a boolean.
    exists = client.check_exists("dummy_id_for_connection_test")
    assert isinstance(exists, bool)
