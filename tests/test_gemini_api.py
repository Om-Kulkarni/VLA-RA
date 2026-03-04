import os
import pytest
from google import genai
from google.genai import types
from dotenv import load_dotenv


def test_gemini_connection():
    load_dotenv()
    """
    Test the Gemini API connection.
    Sends a minimal prompt to use very few tokens.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    assert gemini_key is not None, "GEMINI_API_KEY not set in environment."

    client = genai.Client(api_key=gemini_key)

    # Send a tiny prompt to minimize token usage
    prompt = "Reply with 'OK' and nothing else."

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )
    text = response.text.strip()
    # Just check that it replied with a string, not necessarily strict "OK" to avoid flakiness
    assert isinstance(text, str)
    assert len(text) > 0
