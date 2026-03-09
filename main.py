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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Initializing LangGraph workflow...")
    workflow_app = create_workflow()

    # Read Manifesto to inject into state
    manifesto_content = ""
    try:
        with open("core/manifesto.md", "r") as f:
            manifesto_content = f.read()
    except FileNotFoundError:
        logger.warning("core/manifesto.md not found. Proceeding with empty manifesto.")

    # Define a test initial state
    initial_state = {
        "search_query": "Recent advancements in Vision-Language-Action (VLA) models for bimanual manipulation",
        "manifesto": manifesto_content,
        "scout_results": [],
        "filtered_results": [],
        "analysis_outputs": {},
        "rubric_scores": {},
        "errors": [],
    }

    print(f"Triggering workflow with query: '{initial_state['search_query']}'\n")

    # Execute the graph and pass LangSmith metadata for tracing
    config = {
        "configurable": {"thread_id": "test_workflow"},
        "metadata": {"run_name": "VLA-RA-Testing"},
    }

    # First invocation will run until the analyst interrupt
    logger.info("Starting initial graph execution...")
    paused_state = workflow_app.invoke(initial_state, config=config)

    # Check for interrupt
    snapshot = workflow_app.get_state(config)
    if snapshot.next and "analyst" in snapshot.next:
        print("\n" + "=" * 50)
        print("⏸️  HITL INTERRUPT: Critic Evaluation Complete")
        print("=" * 50)

        rubric_scores = paused_state.get("rubric_scores", {})

        # Display top candidates (score >= 3.0)
        top_candidates = {
            aid: score for aid, score in rubric_scores.items() if score >= 3.0
        }

        if top_candidates:
            print(f"Found {len(top_candidates)} priority candidates (Score >= 3.0):")
            for aid, score in top_candidates.items():
                print(f"  - {aid}: {score:.1f}/5.0")

            print(
                "\nEnter ArXiv IDs to approve for Analyst Deep-Dive (comma-separated), or press Enter to skip:"
            )
            try:
                approved_input = input("> ")
                approved_list = [
                    x.strip() for x in approved_input.split(",") if x.strip()
                ]
            except EOFError:
                approved_list = []

            print(f"\nApproving {len(approved_list)} papers. Resuming workflow...")

            # Update state with approved list
            workflow_app.update_state(config, {"approved_papers": approved_list})
        else:
            print(
                "No candidates scored >= 3.0. analyst node will run with empty approved list."
            )
            workflow_app.update_state(config, {"approved_papers": []})

        # Resume the graph from the paused state
        final_state = workflow_app.invoke(None, config=config)
    else:
        # If it didn't interrupt (e.g., failed earlier), final_state is paused_state
        final_state = paused_state

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

    print("\n\nCritic Final Scores:")
    rubric_scores = final_state.get("rubric_scores", {})
    if not rubric_scores:
        print("No scores calculated or Critic bypassed.")

    for arxiv_id, score in rubric_scores.items():
        print(f"  ArXiv ID {arxiv_id}: Priority Score = {score}/5.0")


if __name__ == "__main__":
    main()
