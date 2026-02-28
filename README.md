# VLA-RA

This project is a LangGraph-based framework for retrieving, parsing, and evaluating papers and code repositories based on a deterministic weighted rubric.

## "Must-Read" Rubric Documentation
*To be defined: A weighted scoring logic applied by the `critic` node.*

## Quickstart

This project uses `uv` as the package manager.

1. Ensure `uv` is installed.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Set your environment variables (see `.env.example`).
4. Run the workflow:
   ```bash
   uv run main.py
   ```
