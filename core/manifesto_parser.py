"""
Manifesto Parser

Extracts structured list data from specific sections of the markdown manifesto.
All agents and tools must read domain-specific config (labs, conferences, etc.)
through this module — never from hardcoded constants in source files.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def parse_section_list(manifesto_text: str, section_heading: str) -> list[str]:
    """
    Extracts a markdown bullet list from a given ## section heading.

    Scans the manifesto for a line matching '## {section_heading}', then collects
    all '- Item' lines until the next ## heading or end of file.

    Args:
        manifesto_text (str): The full text content of core/manifesto.md.
        section_heading (str): The exact heading text to search for (case-insensitive).

    Returns:
        list[str]: Extracted item strings, stripped of leading '- ' and whitespace.
                   Returns an empty list if the section is absent — never raises.
    """
    if not manifesto_text:
        logger.warning(f"manifesto_parser: manifesto_text is empty, cannot parse '{section_heading}'.")
        return []

    # Build a pattern that finds the heading then captures lines until next ## or EOF
    pattern = re.compile(
        r"^##\s+" + re.escape(section_heading) + r"\s*$"  # match the heading
        r"(.*?)"                                             # capture everything after
        r"(?=^##\s|\Z)",                                     # until next ## heading or EOF
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )

    match = pattern.search(manifesto_text)
    if not match:
        logger.warning(
            f"manifesto_parser: Section '## {section_heading}' not found in manifesto. "
            "Returning empty list. Add the section to core/manifesto.md to enable this feature."
        )
        return []

    section_body = match.group(1)

    # Extract '- Item' lines, strip the bullet and surrounding whitespace
    items = [
        line.lstrip("- ").strip()
        for line in section_body.splitlines()
        if line.strip().startswith("- ")
    ]

    if not items:
        logger.warning(
            f"manifesto_parser: Section '## {section_heading}' found but contains no '- Item' lines."
        )

    return items


def parse_priority_labs(manifesto_text: str) -> list[str]:
    """Convenience wrapper. Returns the Priority Labs list from the manifesto."""
    return parse_section_list(manifesto_text, "Priority Labs")


def parse_target_conferences(manifesto_text: str) -> list[str]:
    """Convenience wrapper. Returns the Target Conferences list from the manifesto."""
    return parse_section_list(manifesto_text, "Target Conferences")
