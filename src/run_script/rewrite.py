#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rewrite.py – 呼叫 query_rewrite，將結果輸出為 JSON（stdout）

用法範例
--------
python rewrite.py \
    --question "dose adjustment for CYP2C19 poor metabolizer" \
    --tokens 256 --temperature 1.0 --top-p 0.0 --top-k 1
"""
from __future__ import annotations

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]   # …/src/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import sys
from pathlib import Path

from query_rewrite.cpic_query_rewrite import query_rewrite
from prompt import system_prompt as _DEFAULT_SYS

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser("Run Nemotron query-rewrite on one question")
parser.add_argument("--question", required=True, help="User question")

# optional system prompt
parser.add_argument("--system-prompt-file",
                    help="Path to a txt file; overrides default prompt")

# generation knobs
parser.add_argument("--tokens",   type=int,   default=256,
                    help="num_tokens_to_generate")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-p",       type=float, default=0.0)
parser.add_argument("--top-k",       type=int,   default=1)

# ckpt path override (rarely used)
parser.add_argument("--ckpt-path", help="Override finetuned checkpoint dir")

args = parser.parse_args()

# ----------------------------------------------------------------------
# Resolve system prompt  (file > default)
# ----------------------------------------------------------------------
if args.system_prompt_file:
    sys_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
else:
    sys_prompt = _DEFAULT_SYS

# ----------------------------------------------------------------------
# Call helper
# ----------------------------------------------------------------------
queries = query_rewrite(
    question_prompt=args.question,
    system_prompt=sys_prompt,
    num_tokens_to_generate=args.tokens,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    **({"ckpt_path": args.ckpt_path} if args.ckpt_path else {}),
)

# 統一用 UTF-8 輸出；確保管線時不會亂碼
json.dump(queries, sys.stdout, ensure_ascii=False)
sys.stdout.write("\n")
sys.stdout.flush()
