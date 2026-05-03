"""
Analyst Node

This module serves as the deep-dive reader, utilising DeepSeek V3 via OpenRouter
and Docling to parse and analyse PDFs and extract architecture details.

Research domain and analytical focus are driven entirely by the manifesto in
GraphState — no domain-specific terms are hardcoded here.

Outputs:
    - outputs/{arxiv_id}.md  — 1-page Markdown summary written to disk.
    - data/vla_ra.db         — analysis_summary persisted to metadata column.
"""

import os
import re
import logging
import requests
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from core.config import get_config
from core.database import DatabaseClient
from tools.parser import PDFParserTool
from tools.code_interpreter import CodeInterpreterTool

OUTPUTS_DIR = Path("outputs")

logger = logging.getLogger(__name__)


def analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses full paper texts and generates a deep 1-page summary for each approved paper.
    Persists the analysis summary to the local database and updates status to 'analysed'.

    The analytical focus (what to look for, what matters) is derived from the
    research manifesto — not from hardcoded domain terms.

    Input State:
        - state (Dict[str, Any]): Graph state containing:
            - 'approved_papers': list of arxiv_ids approved at the HITL interrupt.
            - 'filtered_results': full paper metadata from the Librarian.
            - 'manifesto': the central research manifesto.

    Output State:
        - Dict[str, Any]: State subset with 'analysis_outputs' — a dict of
          {arxiv_id: {"title": str, "summary": str}} for successfully analysed papers.
    """
    logger.info("Analyst Node: Starting deep-dive on approved papers.")

    app_config = get_config()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY must be set in environment variables.")

    ai_client = OpenAI(api_key=api_key, base_url=app_config.openrouter_base_url)
    db_client = DatabaseClient(db_path=app_config.db_path)

    parser = PDFParserTool()
    interpreter = CodeInterpreterTool()

    manifesto = state.get("manifesto", "")
    approved_ids = state.get("approved_papers", [])
    papers_to_evaluate = state.get("filtered_results", [])
    if not papers_to_evaluate:
        papers_to_evaluate = state.get("scout_results", [])

    analysis_outputs = state.get("analysis_outputs", {})

    for paper in papers_to_evaluate:
        arxiv_id = paper.get("arxiv_id", "")
        if arxiv_id not in approved_ids:
            continue

        title = paper.get("title", "")
        logger.info(f"Analyst processing approved paper: {title} ({arxiv_id})")

        # 1. Download PDF (if not already downloaded)
        pdf_url = paper.get("url", "").replace("abs", "pdf")
        pdf_path = f"/tmp/{arxiv_id}.pdf"

        if not os.path.exists(pdf_path):
            try:
                res = requests.get(pdf_url, timeout=30)
                if res.status_code == 200:
                    with open(pdf_path, "wb") as f:
                        f.write(res.content)
                else:
                    logger.error(
                        f"PDF download for {arxiv_id} returned HTTP {res.status_code}."
                    )
                    analysis_outputs[arxiv_id] = {
                        "error": f"PDF download failed (HTTP {res.status_code})"
                    }
                    continue
            except Exception as e:
                logger.error(f"Failed to download PDF for {arxiv_id}: {e}")
                analysis_outputs[arxiv_id] = {"error": "PDF download failed"}
                continue

        # 2. Parse PDF to clean Markdown with Docling
        try:
            clean_md = parser.parse_pdf(pdf_path)
        except Exception as e:
            logger.error(f"Failed Docling parse for {arxiv_id}: {e}")
            analysis_outputs[arxiv_id] = {"error": "Docling parse failed"}
            continue

        # 3. Code/repository context
        comment = paper.get("comment", "")
        summary = paper.get("summary", "")
        github_match = re.search(
            r"(https://github\.com/[^\s]+)", comment + " " + summary
        )

        code_context = "No GitHub link found for code preview."
        if github_match:
            repo_url = github_match.group(1)
            logger.info(f"Found repository for {arxiv_id}: {repo_url}")
            repo_data = interpreter.analyze_repository(repo_url)
            code_context = (
                f"GitHub Repo: {repo_url}\n"
                f"README Preview:\n{repo_data.get('readme', '')[:1500]}\n\n"
                f"Requirements Preview:\n{repo_data.get('requirements', '')[:1000]}"
            )

        # 4. Generate Deep Research Brief — focus derived from manifesto, not hardcoded domain
        # Block 4 injects the manifesto's Core Interests so the alignment question is
        # always specific to the active research domain, not a hardcoded topic.
        from core.manifesto_parser import parse_section_list

        core_interests = parse_section_list(manifesto, "Core Interests")
        interests_str = (
            "\n".join(f"- {i}" for i in core_interests) if core_interests else manifesto
        )

        prompt = (
            "You are a Senior Research Lead and Technical Content Creator.\n"
            "Based on the Research Manifesto, produce a 'Deep Research Brief' that is ready for social distribution.\n\n"
            f"Research Manifesto:\n{manifesto}\n\n"
            "Structure your output into four distinct blocks exactly as shown below:\n\n"
            "--- BLOCK 1: THE SOCIAL HOOKS (For X) ---\n"
            "1. **The 'Contrarian' Hook:** A 1-sentence statement that challenges common wisdom based on this paper.\n"
            "2. **The 'Outcome' Hook:** A 1-sentence statement about the most impressive real-world task the paper achieved.\n"
            "3. **The 'Vibe' Hook:** A witty, one-liner (Kache-style) about a technical pain point mentioned "
            "(e.g., 'Another day, another 50,000 GPU hours for a 5% pick-rate boost').\n\n"
            "--- BLOCK 2: THE TECHNICAL MEAT (For Substack/Threads) ---\n"
            "1. **The Architecture Breakthrough:** Explain the VLA/World Model innovation like I'm a fellow PhD "
            "(mention specific loss functions or tokenization methods).\n"
            "2. **The 'PhD Reality Check':** What did they gloss over? "
            "(e.g., 'Zero-shot' but on a fixed background? Latency? Dataset diversity?)\n"
            "3. **Hardware/Code:** Is the URDF/Policy code actually usable? "
            "What does the repo actually provide vs. what's missing?\n\n"
            "--- BLOCK 3: VISUAL ASSETS ---\n"
            "1. **The 'Main Image' Suggestion:** Describe exactly which Figure from the paper I should screenshot for the cover.\n"
            "2. **The 'Demo' Suggestion:** Which 10-second clip from the project site should I screen-record?\n\n"
            "--- BLOCK 4: MANIFESTO ALIGNMENT ---\n"
            "For each of our active Core Interests below, write 1-2 sentences on exactly how this paper advances, "
            "challenges, or is irrelevant to that specific interest. Be blunt — say 'Not relevant' if it isn't.\n\n"
            f"Our Core Interests:\n{interests_str}\n\n"
            f"Paper Title: {title}\n\n"
            "--- FULL PAPER MARKDOWN ---\n"
            f"{clean_md[:50000]}\n\n"  # Context window guard
            "--- GITHUB / CODE CONTEXT ---\n"
            f"{code_context}"
        )

        try:
            response = ai_client.chat.completions.create(
                model=app_config.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_summary = response.choices[0].message.content
            analysis_outputs[arxiv_id] = {"title": title, "summary": generated_summary}
            logger.info(f"Successfully generated summary for {arxiv_id}.")

        except Exception as e:
            logger.error(f"LLM Analyst generation failed for {arxiv_id}: {e}")
            analysis_outputs[arxiv_id] = {"error": "LLM generation failed"}
            continue  # Do not persist a failed analysis to DB

        # 5. Write summary to disk as a Markdown file
        try:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:60]
            output_path = OUTPUTS_DIR / f"{arxiv_id}_{safe_title}.md"
            md_content = (
                f"# {title}\n"
                f"**ArXiv ID:** {arxiv_id}  \n"
                f"**URL:** {paper.get('url', '')}  \n\n"
                f"---\n\n"
                f"{generated_summary}"
            )
            output_path.write_text(md_content, encoding="utf-8")
            logger.info(f"Summary written to: {output_path}")
        except Exception as write_err:
            logger.warning(f"Could not write summary file for {arxiv_id}: {write_err}")

        # 5. Persist analysis summary to DB and update status to "analysed"
        try:
            db_client.update_paper_score(
                external_id=arxiv_id,
                score=None,  # Score already written by Critic — preserve it
                metadata={"analysis_summary": generated_summary},
            )
            db_client.update_status(arxiv_id, "analysed")
            logger.info(
                f"Analysis for {arxiv_id} persisted to DB. Status → 'analysed'."
            )
        except Exception as db_err:
            logger.error(f"Failed to persist analysis for {arxiv_id} to DB: {db_err}")
            raise  # Fail loudly per .agrules

    return {"analysis_outputs": analysis_outputs}
