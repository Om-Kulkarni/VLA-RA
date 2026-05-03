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

    # Agent / non-interactive mode (no stdin prompts)
    uv run main.py --arxiv-id 2601.16163 --auto-approve
    uv run main.py --arxiv-id 2601.16163 --approve-ids 2601.16163 --json
    uv run main.py --arxiv-id 2601.16163 --no-analyst --json

    # Override the score threshold for HITL display
    uv run main.py --arxiv-id 2601.16163 --threshold 2.5 --auto-approve
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
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
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help=(
            "Non-interactive: automatically approve all papers that score at or above "
            "--threshold for Analyst deep-dive. Disables the stdin prompt."
        ),
    )
    parser.add_argument(
        "--approve-ids",
        nargs="+",
        metavar="ID",
        help=(
            "Non-interactive: explicitly approve these ArXiv IDs for Analyst deep-dive. "
            "Overrides --auto-approve. Disables the stdin prompt. "
            "Example: --approve-ids 2601.16163"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        metavar="SCORE",
        help=(
            "Minimum Critic score (0–5) a paper must reach to be shown as a priority "
            "candidate. Default: 3.0. Used by --auto-approve."
        ),
    )
    parser.add_argument(
        "--no-analyst",
        action="store_true",
        help="Stop after the Critic. Score papers but skip the Analyst deep-dive entirely.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help=(
            "Emit a single JSON object to stdout as the final line of output. "
            "Useful for agent callers that parse structured results. "
            "Human-readable output is still printed to stderr."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared HITL + result printing logic
# ---------------------------------------------------------------------------

def resolve_approved_papers(
    paused_state: dict,
    auto_approve: bool,
    approve_ids: list[str] | None,
    threshold: float,
    non_interactive: bool,
) -> list[str]:
    """
    Determines which papers to approve for the Analyst based on CLI flags.

    Priority order:
      1. --approve-ids  → use exactly these IDs (agent-supplied explicit list)
      2. --auto-approve → approve all papers scoring >= threshold automatically
      3. Interactive    → prompt the user via stdin

    Returns a list of arxiv_id strings to approve.
    """
    rubric_scores = paused_state.get("rubric_scores", {})
    sorted_scores = sorted(rubric_scores.items(), key=lambda x: x[1], reverse=True)

    if approve_ids:
        return approve_ids

    if auto_approve:
        return [aid for aid, score in sorted_scores if score >= threshold]

    if non_interactive:
        # Fallback: non-interactive callers that specified neither flag get nothing approved
        return []

    # Interactive: prompt the user
    papers = paused_state.get("filtered_results", []) or paused_state.get("scout_results", [])
    title_map = {p.get("arxiv_id", ""): p.get("title", "Untitled") for p in papers}
    top = [(aid, score) for aid, score in sorted_scores if score >= threshold]

    print(f"\n{'='*60}")
    print("⏸️  HITL INTERRUPT: Critic Evaluation Complete")
    print(f"{'='*60}")

    if top:
        print(f"\nFound {len(top)} priority candidate(s) (Score ≥ {threshold}):\n")
        for aid, score in top:
            print(f"  [{score:.1f}/5.0]  {aid}  —  {title_map.get(aid, 'Unknown Title')}")

    if sorted_scores:
        below = [(aid, score) for aid, score in sorted_scores if score < threshold]
        if below:
            print("\nBelow threshold:")
            for aid, score in below:
                print(f"  [{score:.1f}/5.0]  {aid}  —  {title_map.get(aid, 'Unknown Title')}  (below threshold)")

    print("\nEnter ArXiv IDs to approve for Analyst deep-dive (comma-separated), or press Enter to skip all:")
    try:
        approved_input = input("> ")
        return [x.strip() for x in approved_input.split(",") if x.strip()]
    except EOFError:
        return []


def run_hitl_and_resume(workflow_app, paused_state, config, args, logger):
    """
    Handles the HITL interrupt between Critic and Analyst.
    Respects --auto-approve, --approve-ids, --no-analyst, and --json flags.
    Returns the final state after Analyst completes (or Critic state if --no-analyst).
    """
    snapshot = workflow_app.get_state(config)
    if not (snapshot.next and "analyst" in snapshot.next):
        return paused_state

    non_interactive = args.auto_approve or bool(args.approve_ids) or args.no_analyst

    if args.no_analyst:
        approved_list = []
        if not args.json_output:
            print("\n⏭️  --no-analyst: Skipping Analyst deep-dive.")
    else:
        approved_list = resolve_approved_papers(
            paused_state=paused_state,
            auto_approve=args.auto_approve,
            approve_ids=args.approve_ids,
            threshold=args.threshold,
            non_interactive=non_interactive,
        )
        if not args.json_output:
            mode = "auto" if (args.auto_approve or args.approve_ids) else "manual"
            print(f"\nApproving {len(approved_list)} paper(s) [{mode}]. Resuming workflow...")

    workflow_app.update_state(config, {"approved_papers": approved_list})
    return workflow_app.invoke(None, config=config)


def build_json_result(final_state: dict) -> dict:
    """Builds a structured dict of workflow results suitable for JSON output."""
    papers = final_state.get("filtered_results", []) or final_state.get("scout_results", [])
    title_map = {p.get("arxiv_id", ""): p.get("title", "Untitled") for p in papers}
    rubric_scores = final_state.get("rubric_scores", {})
    analysis_outputs = final_state.get("analysis_outputs", {})

    analyses = []
    for arxiv_id, output in analysis_outputs.items():
        safe_title = re.sub(r'[^\w\s-]', '', output.get('title', arxiv_id)).strip().replace(' ', '_')[:60]
        output_path = str(Path("outputs") / f"{arxiv_id}_{safe_title}.md")
        analyses.append({
            "arxiv_id": arxiv_id,
            "title": output.get("title", ""),
            "output_file": output_path,
            "error": output.get("error"),
            "summary": output.get("summary") if "error" not in output else None,
        })

    return {
        "status": "ok",
        "scores": [
            {
                "arxiv_id": aid,
                "score": score,
                "title": title_map.get(aid, "Unknown"),
            }
            for aid, score in sorted(rubric_scores.items(), key=lambda x: x[1], reverse=True)
        ],
        "analyses": analyses,
        "output_dir": str(Path("outputs").resolve()),
    }


def print_final_results(final_state: dict, json_output: bool = False):
    """Prints human-readable results to stdout. Emits JSON to stdout if --json."""
    # Human-readable block (always printed)
    print(f"\n{'='*60}\nWORKFLOW COMPLETE\n{'='*60}")

    scout_results = final_state.get("scout_results", [])
    if scout_results:
        print("\nScout Results Discovered:")
        for idx, result in enumerate(scout_results, start=1):
            print(f"\n  {idx}. {result.get('title')}")
            print(f"     URL: {result.get('url')}")
    else:
        print("\nScout Results: (Express mode — Scout skipped)")

    filtered_results = final_state.get("filtered_results", [])
    print("\nLibrarian Filtered Results:")
    if not filtered_results:
        print("  No papers passed the Librarian's filters.")
    for idx, result in enumerate(filtered_results, start=1):
        print(f"\n  {idx}. [{result.get('arxiv_id')}]  {result.get('title')}")
        print(f"     URL: {result.get('url')}")

    rubric_scores = final_state.get("rubric_scores", {})
    papers = filtered_results or scout_results
    title_map = {p.get("arxiv_id", ""): p.get("title", "Untitled") for p in papers}
    print("\nCritic Final Scores:")
    if not rubric_scores:
        print("  No scores calculated.")
    for aid, score in sorted(rubric_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  [{score:.1f}/5.0]  {aid}  —  {title_map.get(aid, 'Unknown')}")

    analysis_outputs = final_state.get("analysis_outputs", {})
    print("\nAnalyst Deep-Dive Summaries:")
    if not analysis_outputs:
        print("  No papers were approved for deep-dive analysis.")
    for arxiv_id, output in analysis_outputs.items():
        if "error" in output:
            print(f"\n  [{arxiv_id}] ERROR: {output['error']}")
        else:
            safe_title = re.sub(r'[^\w\s-]', '', output.get('title', arxiv_id)).strip().replace(' ', '_')[:60]
            output_path = Path("outputs") / f"{arxiv_id}_{safe_title}.md"
            print(f"\n{'='*60}")
            print(f"  {output.get('title', arxiv_id)}")
            print(f"  ArXiv ID : {arxiv_id}")
            print(f"  Saved to : {output_path}")
            print(f"{'='*60}")
            print(output.get("summary", ""))

    # JSON block — emitted as the very last line so agents can read it cleanly
    if json_output:
        result = build_json_result(final_state)
        print("\n--- JSON_RESULT ---")
        print(json.dumps(result, indent=2))
        print("--- END_JSON_RESULT ---")


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

    # In --json mode, redirect INFO logs to stderr so stdout stays clean for JSON
    log_stream = sys.stderr if args.json_output else sys.stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=log_stream,
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
        if not args.json_output:
            print(f"\n🚀 Express mode — loading {len(args.arxiv_id)} paper(s) directly from ArXiv.\n")

        papers = fetch_papers_by_id(args.arxiv_id, logger)
        insert_express_papers(papers, app_config.db_path, logger)

        if not args.json_output:
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
            "filtered_results": papers,
            "analysis_outputs": {},
            "rubric_scores": {},
            "errors": [],
        }

        logger.info("Express mode: invoking Critic directly...")
        paused_state = workflow_app.invoke(initial_state, config=config)
        final_state = run_hitl_and_resume(workflow_app, paused_state, config, args, logger)
        print_final_results(final_state, json_output=args.json_output)
        return

    # -----------------------------------------------------------------------
    # FULL PIPELINE MODE
    # -----------------------------------------------------------------------
    if not args.json_output:
        print("\nTriggering full workflow. Research direction is defined in core/manifesto.md\n")

    workflow_app = create_workflow()
    config = {
        "configurable": {"thread_id": "full_workflow"},
        "metadata": {"run_name": "VLA-RA-Full"},
    }

    initial_state = {
        # SINGLE POINT OF EDIT: edit core/manifesto.md to change research direction.
        # search_query is an optional one-off narrowing hint — leave empty normally.
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
    final_state = run_hitl_and_resume(workflow_app, paused_state, config, args, logger)
    print_final_results(final_state, json_output=args.json_output)


if __name__ == "__main__":
    main()
