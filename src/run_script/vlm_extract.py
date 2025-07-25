#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_page.py  ──  CLI wrapper around send_pdf_page()

Example
-------
conda run -n vespa_env python extract_page.py \
    --pdf "/path/to/guideline.pdf" \
    --page-idx 1 \
    --prompt "Summarise the tables on this page."
"""

from __future__ import annotations

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]   # …/src/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# send_pdf_page is the function we previously wrote in cpic_vlm_parse:

from cpic_vlm_parse.extract_api_call import send_pdf_page



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Send a single PDF page to NVIDIA VLM API")
    p.add_argument("--pdf", required=True, help="Local PDF file path")
    p.add_argument("--page-idx", type=int, default=0,
                   help="0-based page index (default: 0)")
    p.add_argument("--prompt", required=True, help="User prompt / task")
    p.add_argument("--model",
                   default="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
                   help="Model ID served by NVIDIA integrate API")
    p.add_argument("--api-key", default=None,
                   help="If omitted, reads NVIDIA_API_TOKEN env-var")
    # advanced sampling flags (optional)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=1024)
    return p


def main() -> None:
    load_dotenv()  # Load environment variables from .env file
    args = build_arg_parser().parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        sys.exit(f"❌  PDF not found: {pdf_path}")

    api_key = args.api_key or os.getenv("NVIDIA_API_TOKEN")
    if not api_key:
        sys.exit("❌  Provide --api-key or set NVIDIA_API_TOKEN environment var.")

    # --- Call helper ----------------------------------------------------
    send_pdf_page(
        pdf_file=str(pdf_path),
        page_idx=args.page_idx,
        user_prompt=args.prompt,
        model=args.model,
        api_key=api_key,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
