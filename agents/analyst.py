"""
Analyst Node

This module serves as the multimodal reader, utilising DeepSeek V3 via OpenRouter
and Docling to parse and analyse PDFs and extract architecture details.
"""

import os
import re
import json
import logging
import requests
from typing import Any, Dict

from openai import OpenAI

from core.config import get_config
from tools.parser import PDFParserTool
from tools.code_interpreter import CodeInterpreterTool

logger = logging.getLogger(__name__)


def analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses full paper texts/images and extracts robotics-specific data representations.
    DeepSeek V3 is prompted with structured Markdown content for architecture analysis.

    Input State:
        - state (Dict[str, Any]): Graph state containing downloaded paper paths and 'filtered_results'.

    Output State:
        - Dict[str, Any]: State subset with 'analysis_outputs' representing structural extractions.
    """
    logger.info("Analyst Node: Starting multimodal deep dive on approved papers.")

    app_config = get_config()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY must be set in environment variables.")

    ai_client = OpenAI(api_key=api_key, base_url=app_config.openrouter_base_url)

    parser = PDFParserTool()
    interpreter = CodeInterpreterTool()

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
            except Exception as e:
                logger.error(f"Failed to download PDF for {arxiv_id}: {e}")
                analysis_outputs[arxiv_id] = {"error": "PDF download failed"}
                continue

        # 2. Parse PDF to Clean Markdown with Docling
        try:
            clean_md = parser.parse_pdf(pdf_path)
        except Exception as e:
            logger.error(f"Failed Docling parse for {arxiv_id}: {e}")
            analysis_outputs[arxiv_id] = {"error": "Docling parse failed"}
            continue

        # 3. Code Implementation Preview
        comment = paper.get("comment", "")
        summary = paper.get("summary", "")
        github_match = re.search(
            r"(https://github\.com/[^\s]+)", comment + " " + summary
        )

        code_context = "No Github link found for code preview."
        if github_match:
            repo_url = github_match.group(1)
            logger.info(f"Found repository: {repo_url}")
            repo_data = interpreter.analyze_repository(repo_url)
            code_context = (
                f"GitHub Repo: {repo_url}\n"
                f"README Preview:\n{repo_data.get('readme', '')[:1500]}\n\n"
                f"Requirements Preview:\n{repo_data.get('requirements', '')[:1000]}"
            )

        # 4. Generate the 1-page summary
        prompt = (
            "You are an elite Robotics Research Analyst.\n"
            "Please provide a deep, 1-page Markdown summary of the following research paper, "
            "focusing heavily on:\n"
            "1. The Core Research and Outcomes\n"
            "2. Technical Implementation Details (Neural net architectures, loss functions, action spaces)\n"
            "3. Real-world testing, hardware setups, and sim-to-real gaps.\n"
            "4. Implementation preview (steps to get it running based on provided repo context).\n\n"
            "Skip all marketing fluff. Use bullet points and bolding for readability.\n\n"
            f"Paper Title: {title}\n\n"
            "--- FULL PAPER MARKDOWN ---\n"
            f"{clean_md[:50000]}\n\n"  # Limiting context window slightly for safety
            "--- GITHUB CONTEXT ---\n"
            f"{code_context}"
        )

        try:
            response = ai_client.chat.completions.create(
                model=app_config.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            analysis_outputs[arxiv_id] = {
                "title": title,
                "summary": response.choices[0].message.content,
            }
            logger.info(f"Successfully analyzed {arxiv_id}")
        except Exception as e:
            logger.error(f"LLM Analyst generation failed for {arxiv_id}: {e}")
            analysis_outputs[arxiv_id] = {"error": "LLM Generation failed"}

    return {"analysis_outputs": analysis_outputs}
