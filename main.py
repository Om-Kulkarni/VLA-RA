"""
Main Entry Point

This is the primary runtime script that initializes the environment and triggers
the LangGraph workflow. Based on the UV runtime policy, invoke via `uv run main.py`.

Usage:
    # Full pipeline (Scout → Librarian → Critic → Analyst)
    uv run main.py

    # Express mode — skip Scout/Librarian, go straight to Critic + Analyst
    uv run main.py --arxiv-id 2601.16163
    uv run main.py --arxiv-id 2601.16163 2412.00001   # multiple papers
"""

import os
import argparse
import logging
from dotenv import load_dotenv
from chains.graph import create_workflow, create_express_workflow


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLA-RA: Vision-Language-Action Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arxiv-id",
        nargs="+",
        metavar="ID",
        help=(
            "Express mode: one or more ArXiv IDs to critique and analyse directly, "
            "skipping the Scout and Librarian pipeline. "
            "Example: --arxiv-id 2601.16163"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared HITL + result printing logic
# ---------------------------------------------------------------------------

def run_hitl_and_resume(workflow_app, paused_state, config, logger):
    """
    Handles the HITL interrupt between Critic and Analyst:
    - Displays scored papers with their titles
    - Prompts the user to approve papers for deep-dive
    - Resumes the graph with the approved list
    Returns the final state after Analyst completes.
    """
    snapshot = workflow_app.get_state(config)
    if not (snapshot.next and "analyst" in snapshot.next):
        # Workflow didn't reach the interrupt (e.g. failed earlier)
        return paused_state

    print("\n" + "=" * 60)
    print("⏸️  HITL INTERRUPT: Critic Evaluation Complete")
    print("=" * 60)

    rubric_scores = paused_state.get("rubric_scores", {})

    # Build arxiv_id → title lookup from filtered_results (or scout_results fallback)
    papers = paused_state.get("filtered_results", []) or paused_state.get("scout_results", [])
    title_map = {p.get("arxiv_id", ""): p.get("title", "Untitled") for p in papers}

    # Show all scored papers sorted by score descending
    sorted_scores = sorted(rubric_scores.items(), key=lambda x: x[1], reverse=True)

    top_candidates = [(aid, score) for aid, score in sorted_scores if score >= 3.0]

    if top_candidates:
        print(f"\nFound {len(top_candidates)} priority candidates (Score ≥ 3.0):\n")
        for aid, score in top_candidates:
            title = title_map.get(aid, "Unknown Title")
            print(f"  [{score:.1f}/5.0]  {aid}  —  {title}")

        print("\nAll scored papers:")
        for aid, score in sorted_scores:
            if score < 3.0:
                title = title_map.get(aid, "Unknown Title")
                print(f"  [{score:.1f}/5.0]  {aid}  —  {title}  (below threshold)")

        print(
            "\nEnter ArXiv IDs to approve for Analyst deep-dive (comma-separated), "
            "or press Enter to skip all:"
        )
        try:
            approved_input = input("> ")
            approved_list = [x.strip() for x in approved_input.split(",") if x.strip()]
        except EOFError:
            approved_list = []

        print(f"\nApproving {len(approved_list)} paper(s). Resuming workflow...")
        workflow_app.update_state(config, {"approved_papers": approved_list})
    else:
        print("\nNo candidates scored ≥ 3.0.")
        if sorted_scores:
            print("\nAll scored papers:")
            for aid, score in sorted_scores:
                title = title_map.get(aid, "Unknown Title")
                print(f"  [{score:.1f}/5.0]  {aid}  —  {title}")
        workflow_app.update_state(config, {"approved_papers": []})

    return workflow_app.invoke(None, config=config)


def print_final_results(final_state: dict):
    """Prints a formatted summary of the complete workflow results."""
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)

    print("\nScout Results Discovered:")
    scout_results = final_state.get("scout_results", [])
    if not scout_results:
        print("  (Express mode — Scout skipped)")
    for idx, result in enumerate(scout_results, start=1):
        print(f"\n  Result {idx}: {result.get('title')}")
        print(f"  URL: {result.get('url')}")
        print(f"  Authors: {', '.join(result.get('authors', []))}")

    print("\nLibrarian Filtered Results:")
    filtered_results = final_state.get("filtered_results", [])
    if not filtered_results:
        print("  No papers passed the Librarian's filters.")
    for idx, result in enumerate(filtered_results, start=1):
        print(f"\n  {idx}. [{result.get('arxiv_id')}]  {result.get('title')}")
        print(f"     URL: {result.get('url')}")

    print("\nCritic Final Scores:")
    rubric_scores = final_state.get("rubric_scores", {})
    if not rubric_scores:
        print("  No scores calculated or Critic bypassed.")
    papers = final_state.get("filtered_results", []) or final_state.get("scout_results", [])
    title_map = {p.get("arxiv_id", ""): p.get("title", "Untitled") for p in papers}
    for aid, score in sorted(rubric_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  [{score:.1f}/5.0]  {aid}  —  {title_map.get(aid, 'Unknown')}")

    print("\n\nAnalyst Deep-Dive Summaries:")
    analysis_outputs = final_state.get("analysis_outputs", {})
    if not analysis_outputs:
        print("  No papers were approved for deep-dive analysis.")
    for arxiv_id, output in analysis_outputs.items():
        if "error" in output:
            print(f"\n  [{arxiv_id}] ERROR: {output['error']}")
        else:
            import re
            from pathlib import Path
            safe_title = re.sub(r'[^\w\s-]', '', output.get('title', arxiv_id)).strip().replace(' ', '_')[:60]
            output_path = Path("outputs") / f"{arxiv_id}_{safe_title}.md"
            print(f"\n{'=' * 60}")
            print(f"  {output.get('title', arxiv_id)}")
            print(f"  ArXiv ID : {arxiv_id}")
            print(f"  Saved to : {output_path}")
            print(f"{'=' * 60}")
            print(output.get("summary", ""))



# ---------------------------------------------------------------------------
# Express mode helpers
# ---------------------------------------------------------------------------

def fetch_papers_by_id(arxiv_ids: list[str], logger) -> list[dict]:
    """Fetches paper metadata from ArXiv for a list of IDs."""
    from tools.arxiv_api import ArxivTool
    tool = ArxivTool()
    papers = []
    for arxiv_id in arxiv_ids:
        logger.info(f"Express mode: fetching ArXiv paper {arxiv_id}...")
        paper = tool.fetch_by_id(arxiv_id)
        papers.append(paper)
        logger.info(f"Fetched: '{paper['title']}'")
    return papers


def insert_express_papers(papers: list[dict], db_path: str, logger):
    """
    Inserts express-mode papers into the DB so the Critic's UPDATE can find them.
    Skips papers that already exist (idempotent).
    """
    from core.database import DatabaseClient
    db = DatabaseClient(db_path=db_path)
    for paper in papers:
        arxiv_id = paper.get("arxiv_id", "")
        if db.check_exists(arxiv_id):
            logger.info(f"Express mode: {arxiv_id} already in DB, skipping insert.")
            continue
        db.insert_record({
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "summary": paper.get("summary", ""),
            "pdf_url": paper.get("url", ""),
            "status": "pending_review",
        })
        logger.info(f"Express mode: inserted {arxiv_id} into DB.")
    db.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point to trigger the LangGraph workflow."""
    load_dotenv()
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Read Manifesto — single source of research direction
    manifesto_content = ""
    try:
        with open("core/manifesto.md", "r") as f:
            manifesto_content = f.read()
    except FileNotFoundError:
        logger.warning("core/manifesto.md not found. Proceeding with empty manifesto.")

    from core.config import get_config
    app_config = get_config()

    # -----------------------------------------------------------------------
    # EXPRESS MODE: --arxiv-id supplied
    # -----------------------------------------------------------------------
    if args.arxiv_id:
        print(f"\n🚀 Express mode — loading {len(args.arxiv_id)} paper(s) directly from ArXiv.\n")

        papers = fetch_papers_by_id(args.arxiv_id, logger)
        insert_express_papers(papers, app_config.db_path, logger)

        for p in papers:
            print(f"  ✓  [{p['arxiv_id']}]  {p['title']}")

        workflow_app = create_express_workflow()
        config = {
            "configurable": {"thread_id": f"express_{'_'.join(args.arxiv_id)}"},
            "metadata": {"run_name": "VLA-RA-Express"},
        }

        initial_state = {
            "search_query": "",
            "manifesto": manifesto_content,
            "scout_results": [],
            "filtered_results": papers,   # pre-loaded — Critic reads from here
            "analysis_outputs": {},
            "rubric_scores": {},
            "errors": [],
        }

        logger.info("Express mode: invoking Critic directly...")
        paused_state = workflow_app.invoke(initial_state, config=config)
        final_state = run_hitl_and_resume(workflow_app, paused_state, config, logger)
        print_final_results(final_state)
        return

    # -----------------------------------------------------------------------
    # FULL PIPELINE MODE
    # -----------------------------------------------------------------------
    print("\nTriggering full workflow. Research direction is defined in core/manifesto.md\n")

    workflow_app = create_workflow()
    config = {
        "configurable": {"thread_id": "full_workflow"},
        "metadata": {"run_name": "VLA-RA-Full"},
    }

    initial_state = {
        # -----------------------------------------------------------------------
        # SINGLE POINT OF EDIT: to change what this system researches, edit
        # core/manifesto.md ONLY. search_query is an optional one-off narrowing
        # hint — leave empty to let the manifesto fully determine direction.
        # -----------------------------------------------------------------------
        "search_query": "",
        "manifesto": manifesto_content,
        "scout_results": [],
        "filtered_results": [],
        "analysis_outputs": {},
        "rubric_scores": {},
        "errors": [],
    }

    logger.info("Starting full pipeline execution...")
    paused_state = workflow_app.invoke(initial_state, config=config)
    final_state = run_hitl_and_resume(workflow_app, paused_state, config, logger)
    print_final_results(final_state)


if __name__ == "__main__":
    main()
