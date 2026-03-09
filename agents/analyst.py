"""
Analyst Node

This module serves as the multimodal reader, utilizing Gemini 1.5 Pro and Docling
to parse and analyze PDFs and extract architecture details.
"""

from typing import Any, Dict
import os
import re
import requests
import logging
from google import genai
from core.config import get_config
from tools.parser import PDFParserTool
from tools.code_interpreter import CodeInterpreterTool

logger = logging.getLogger(__name__)


def analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses full paper texts/images and extracts robotics-specific data representations.
    Gemini 1.5 Pro is explicitly passed image URI references for architecture diagrams.

    Input State:
        - state (Dict[str, Any]): Graph state containing downloaded paper paths and 'filtered_results'.

    Output State:
        - Dict[str, Any]: State subset with 'analysis_outputs' representing structural extractions.
    """
    logger.info("Analyst Node: Starting multimodal deep dive on approved papers.")
    app_config = get_config()
    gemini_key = os.environ.get("GEMINI_API_KEY")
    ai_client = genai.Client(api_key=gemini_key)

    parser = PDFParserTool()
    interpreter = CodeInterpreterTool()

    approved_ids = state.get("approved_papers", [])
    papers_to_evaluate = state.get("filtered_results", [])
    # Fallback to scout_results if filtering was skipped
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
        # Use simple heuristic to find a github link in comment or summary
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
            code_context = f"GitHub Repo: {repo_url}\nREADME Preview:\n{repo_data.get('readme', '')[:1500]}\n\nRequirements Preview:\n{repo_data.get('requirements', '')[:1000]}"

        # 4. Generate the 1-page summary
        prompt = f"""
        You are an elite Robotics Research Analyst.
        Please provide a deep, 1-page Markdown summary of the following research paper, focusing heavily on:
        1. The Core Research and Outcomes
        2. Technical Implementation Details (Neural net architectures, loss functions, action spaces)
        3. Real-world testing, hardware setups, and sim-to-real gaps.
        4. Implementation preview (steps to get it running based on provided repo context).
        
        Skip all marketing fluff. Use bullet points and bolding for readability.
        
        Paper Title: {title}
        
        --- FULL PAPER MARKDOWN ---
        {clean_md[:50000]} # Limiting context window slightly for safety
        
        --- GITHUB CONTEXT ---
        {code_context}
        """

        try:
            response = ai_client.models.generate_content(
                model=app_config.llm_model,
                contents=prompt,
            )
            analysis_outputs[arxiv_id] = {"title": title, "summary": response.text}
            logger.info(f"Successfully analyzed {arxiv_id}")
        except Exception as e:
            logger.error(f"LLM Analyst generation failed for {arxiv_id}: {e}")
            analysis_outputs[arxiv_id] = {"error": "LLM Generation failed"}

    return {"analysis_outputs": analysis_outputs}
