#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vespa_query.py ── run cpic_query() from the CLI and dump results as JSON.

Typical call from main pipeline (inside *vespa_env*):

    conda run -n vespa_env python vespa_query.py \
        --query "dose adjustment for CYP2C19 poor metabolizer" \
        --endpoint-file vespa_endpoint.txt \
        --topk 5 \
        --save-dir retrieved_pages \
        --resize 640
"""
from __future__ import annotations

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]   # …/src/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path

from cpic_vlm_vector_store.cpic_pagewise_query import cpic_query

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser("Run CPIC Vespa retrieval and save images")
parser.add_argument("--query", required=True, help="Natural-language query")

grp = parser.add_mutually_exclusive_group(required=True)
grp.add_argument("--endpoint-file", help="Text file created by pipeline")
grp.add_argument("--endpoint",       help="Direct Vespa URL (skip file)")

parser.add_argument("--topk", type=int, default=3, help="hits to return")
parser.add_argument("--save-dir", default="retrieved_pages",
                    help="where PNG pages are stored")
parser.add_argument("--resize", type=int, default=640,
                    help="short-side resize (None keeps original)")

args = parser.parse_args()

# ----------------------------------------------------------------------
# Call library helper
# ----------------------------------------------------------------------
files = cpic_query(
    query=args.query,
    endpoint_file=args.endpoint_file,
    endpoint=args.endpoint,
    top_k=args.topk,
    save_dir=Path(args.save_dir),
    resize=args.resize,
)

print(json.dumps(files, ensure_ascii=False))
