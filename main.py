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
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing LangGraph workflow...")
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
    
    # Execute the graph and pass LangSmith metadata for tracing
    config = {"configurable": {"thread_id": "test_workflow"}, "metadata": {"run_name": "VLA-RA-Testing"}}
    final_state = workflow_app.invoke(initial_state, config=config)
    
    print("\n--- Workflow Execution Complete ---")
    print("\nScout Results Discovered:")
    
    scout_results = final_state.get("scout_results", [])
    if not scout_results:
        print("No Scout results found.")
        
    for idx, result in enumerate(scout_results, start=1):
        print(f"\nResult {idx}:")
        print(f"  Title: {result.get('title')}")
        print(f"  URL: {result.get('url')}")
        print(f"  Authors: {', '.join(result.get('authors', []))}")
        print(f"  Source Query: {result.get('source_query')}")
        
    print("\n\nLibrarian Filtered Results:")
    filtered_results = final_state.get("filtered_results", [])
    if not filtered_results:
        print("No papers passed the Librarian's filters.")
        
    for idx, result in enumerate(filtered_results, start=1):
        print(f"\nFiltered Result {idx}:")
        print(f"  Title: {result.get('title')}")
        print(f"  ArXiv ID: {result.get('arxiv_id')}")
        print(f"  URL: {result.get('url')}")

if __name__ == "__main__":
    main()
