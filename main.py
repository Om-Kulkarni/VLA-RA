"""
Main Entry Point

This is the primary runtime script that initializes the environment and triggers
the LangGraph workflow. Based on the UV runtime policy, invoke via `uv run main.py`.
"""
import os
import json
from dotenv import load_dotenv
from chains.graph import create_workflow

def main():
    """
    Entry point to trigger the LangGraph workflow.
    """
    # Load environment variables (e.g., GEMINI_API_KEY)
    load_dotenv()
    
    print("Initializing LangGraph workflow...")
    workflow_app = create_workflow()
    
    # Define a test initial state
    initial_state = {
        "search_query": "Recent advancements in Vision-Language-Action (VLA) models for bimanual manipulation",
        "scout_results": [],
        "filtered_results": [],
        "analysis_outputs": {},
        "rubric_scores": {},
        "errors": []
    }
    
    print(f"Triggering workflow with query: '{initial_state['search_query']}'\n")
    
    # Execute the graph
    try:
        final_state = workflow_app.invoke(initial_state)
        
        print("\n--- Workflow Execution Complete ---")
        print("\nScout Results Discovered:")
        
        results = final_state.get("scout_results", [])
        if not results:
            print("No results found.")
            
        for idx, result in enumerate(results, start=1):
            print(f"\nResult {idx}:")
            print(f"  Title: {result.get('title')}")
            print(f"  URL: {result.get('url')}")
            print(f"  Authors: {', '.join(result.get('authors', []))}")
            print(f"  Source Query: {result.get('source_query')}")
            
    except Exception as e:
        print(f"\nWorkflow failed with error: {e}")

if __name__ == "__main__":
    main()
