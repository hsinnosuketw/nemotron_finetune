#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retrieve CPIC guideline pages from Vespa (tutorial‑style API).

The script follows the query pattern shown in pyvespa’s
`docs/sphinx/source/query.ipynb`:

1.  Build the query tensor (`input.query(qt)`).
2.  Compose a **single JSON body** with `yql`, `hits`, `ranking`, etc.
3.  POST it via `app.query(body=query_body)`.
4.  Save the returned page images.

Example
-------
```bash
python retrieve_cpic.py \
  --query "dose adjustment for CYP2C19 poor metabolizer" \
  --endpoint-file vespa_endpoint.txt \
  --top-k 3
```
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import ast
from typing import List

import torch
from tqdm import tqdm
from vespa.application import Vespa
from vespa.io import VespaQueryResponse

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from vespa_setup_pipeline import load_model_and_processor  # same dir import
import pdf_helper

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def build_query_tensor(query: str, model: ColQwen2_5, proc: ColQwen2_5_Processor) -> dict:
    """Return {patch_idx: 128‑d vector} as ordinary Python lists (JSON‑safe)."""
    device = next(model.parameters()).device
    batch = proc.process_queries([query])
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        emb = model(**batch)[0].cpu()           # (patch, 128)
    return {i: v.tolist() for i, v in enumerate(emb)}


def query_vespa(
    app: Vespa,
    user_query: str,
    tensor: dict,
    k: int,
) -> VespaQueryResponse:
    """Run a single query using the notebook‑style JSON body."""
    body = {
        "yql": (
            "select name, path, image, page_number "
            "from pdf_page where userInput(@userQuery)"
        ),
        "hits": k,
        "ranking": "default",
        "timeout": 120,
        "userQuery": user_query,
        "input.query(qt)": tensor,
    }
    return app.query(body=body)


def save_hits(
    hits: list,
    out_dir: Path,
    resize: int | None,
) -> List[str]:
    """Save each hit’s page image as PNG and return file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for rank, hit in enumerate(hits):
        pdf_path = hit["fields"]["path"]
        page = hit["fields"]["page_number"]
        title = Path(pdf_path).stem

        img = pdf_helper.open_pdf_page(pdf_path, page)
        if resize:
            img = pdf_helper.resize_image(img, resize)

        fname = f"{rank:02d}_{title}_p{page+1}.png"
        img.save(out_dir / fname)
        saved.append(str(out_dir / fname))
    return saved

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Retrieve CPIC pages via Vespa")
    p.add_argument("--query", required=True)

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--endpoint", help="Full Vespa URL")
    grp.add_argument("--endpoint-file", help="File containing the URL")

    p.add_argument("--model-name", default="vidore/colqwen2.5-v0.2")
    p.add_argument("--cache-dir", default="/tmp/colqwen_cache")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--save-dir", default="retrieved_pages")
    p.add_argument("--resize", type=int, default=640)
    return p


def main() -> None:
    args = parser().parse_args()

    # Resolve endpoint ------------------------------------------
    if args.endpoint:
        endpoint = args.endpoint
        cert_paths = None  # 若用户直接提供 URL，则不使用证书，或由 Vespa 客户端自行处理
    else:
        endpoint_data = Path(args.endpoint_file).read_text()
        endpoint_dict = ast.literal_eval(endpoint_data)

        # 兼容大小写或不同键名
        endpoint = (
            endpoint_dict.get("Endpoint")
            or endpoint_dict.get("URL")
            or endpoint_dict.get("url")
        )

        cert_paths = (
            endpoint_dict.get("Cert"),
            endpoint_dict.get("Key"),
        )

    # Load model -------------------------------------------------
    device = args.device if torch.cuda.is_available() else "cpu"
    model, proc = load_model_and_processor(
        args.model_name, args.cache_dir, device)

    # Build tensor & query --------------------------------------
    tensor = build_query_tensor(args.query, model, proc)
    app = Vespa(url=endpoint, cert=cert_paths)

    resp = query_vespa(app, args.query, tensor, args.top_k)
    if not resp.is_successful():
        raise RuntimeError(resp.get_error_message())

    paths = save_hits(resp.hits, Path(args.save_dir), args.resize)
    print("Saved files:")
    for p in paths:
        print(" -", p)


if __name__ == "__main__":
    main()
