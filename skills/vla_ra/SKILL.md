---
name: "VLA Research Assistant (VLA-RA)"
description: "Use this skill to research, score, and deeply analyse academic papers on Vision-Language-Action models, robotics, and related AI topics. Supports fetching papers directly by ArXiv ID (express mode) or running a full autonomous discovery pipeline. Always use this skill when the user mentions a paper, ArXiv ID, or asks to research a robotics/AI topic."
metadata:
  openclaw:
    requires:
      bins: ["uv"]
      cwd: "/home/om-kulkarni/Projects/VLA-RA"
---

# VLA-RA Skill: Research, Score, and Analyse Papers

## What This Tool Does

VLA-RA is an autonomous research pipeline that:
1. **Fetches** paper metadata from ArXiv
2. **Scores** papers using a multi-dimensional rubric (relevance, novelty, code maturity, lab prestige)
3. **Generates a Deep Research Brief** — social hooks, technical deep-dive, PhD reality check, and manifesto alignment

All research direction is controlled by `core/manifesto.md`. Do NOT hardcode topics in your commands.

---

## Tool Location

```
Working directory: /home/om-kulkarni/Projects/VLA-RA
Command: uv run main.py [flags]
```

---

## Usage Patterns

### Pattern 1 — Score + Analyse a specific paper (most common)

The user shares a paper link, title, or ArXiv ID. Use this to fetch, score, and generate the full brief.

```bash
cd /home/om-kulkarni/Projects/VLA-RA && uv run main.py \
  --arxiv-id <ARXIV_ID> \
  --auto-approve \
  --json
```

**Output:** Prints human-readable output, then emits a JSON block delimited by `--- JSON_RESULT ---` and `--- END_JSON_RESULT ---`. Parse everything between those delimiters.

**Example — user says "analyse this paper: arXiv:2601.16163":**
```bash
cd /home/om-kulkarni/Projects/VLA-RA && uv run main.py --arxiv-id 2601.16163 --auto-approve --json
```

---

### Pattern 2 — Score only, no deep analysis (fast triage)

Use when the user asks "is this paper relevant?" or "score this paper" without wanting the full brief.

```bash
cd /home/om-kulkarni/Projects/VLA-RA && uv run main.py \
  --arxiv-id <ARXIV_ID> \
  --no-analyst \
  --json
```

---

### Pattern 3 — Multiple papers at once

```bash
cd /home/om-kulkarni/Projects/VLA-RA && uv run main.py \
  --arxiv-id <ID1> <ID2> <ID3> \
  --auto-approve \
  --json
```

---

### Pattern 4 — Score all, but only analyse specific ones

Use when the user wants to triage several papers but only deep-dive the top one.

```bash
cd /home/om-kulkarni/Projects/VLA-RA && uv run main.py \
  --arxiv-id <ID1> <ID2> <ID3> \
  --approve-ids <ID1> \
  --json
```

---

### Pattern 5 — Lower the threshold (for newer papers with few citations)

New papers score lower on citations. Use `--threshold 2.0` if the user specifically requests analysis of a very recent paper.

```bash
cd /home/om-kulkarni/Projects/VLA-RA && uv run main.py \
  --arxiv-id <ARXIV_ID> \
  --threshold 2.0 \
  --auto-approve \
  --json
```

---

## Extracting ArXiv IDs

ArXiv IDs appear in these formats — extract just the numeric part:
- `arXiv:2601.16163` → `2601.16163`
- `https://arxiv.org/abs/2601.16163` → `2601.16163`
- `https://arxiv.org/pdf/2601.16163v1` → `2601.16163`
- `https://arxiv.org/abs/2601.16163v2` → `2601.16163`

If the user gives you a paper title without an ID, use your web search to find the ArXiv ID first.

---

## Parsing the JSON Output

After the run, locate the block between `--- JSON_RESULT ---` and `--- END_JSON_RESULT ---`. It has this schema:

```json
{
  "status": "ok",
  "scores": [
    {
      "arxiv_id": "2601.16163",
      "score": 3.21,
      "title": "Cosmos Policy: Fine-Tuning Video Models..."
    }
  ],
  "analyses": [
    {
      "arxiv_id": "2601.16163",
      "title": "Cosmos Policy: Fine-Tuning Video Models...",
      "output_file": "/home/om-kulkarni/Projects/VLA-RA/outputs/2601.16163_Cosmos_Policy....md",
      "error": null,
      "summary": "# Research Summary: ..."
    }
  ],
  "output_dir": "/home/om-kulkarni/Projects/VLA-RA/outputs"
}
```

- `scores` — all papers scored, sorted by score descending. Present this table to the user.
- `analyses[].output_file` — absolute path to the saved `.md` brief. Tell the user this path.
- `analyses[].summary` — the full Deep Research Brief text. Quote key sections to the user.

---

## Changing Research Domain

To pivot the entire pipeline to a different research field (e.g., drug discovery), edit:
```
/home/om-kulkarni/Projects/VLA-RA/core/manifesto.md
```
Then re-run. No code changes needed.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `OPENROUTER_API_KEY` error | Check `/home/om-kulkarni/Projects/VLA-RA/.env` has the key set |
| Paper not found on ArXiv | Verify the ID is correct; try web search for the canonical ArXiv link |
| Score suspiciously low | Paper may be very new (0 citations). Use `--threshold 2.0` |
| `uv` not found | Run `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| DB locked | Delete `data/vla_ra.db` to reset state (all scores lost) |
