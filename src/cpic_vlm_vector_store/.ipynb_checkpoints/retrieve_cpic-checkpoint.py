#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI for retrieving CPIC guideline pages via Vespa.

Usage:
    python retrieve_cpic.py \
        --query "your question" \
        --endpoint https://your-app-endpoint/ \
        [--model-name MODEL] [--cache-dir DIR] [--device DEVICE] \
        [--top-k N] [--save-dir DIR] [--resize PIXELS]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from vespa.application import Vespa
from vespa.io import VespaQueryResponse

from vespa_setup_pipeline import load_model_and_processor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import pdf_helper


def retrieve_cpic_pages(
    query: str,
    model: ColQwen2_5,
    processor: ColQwen2_5_Processor,
    vespa_app: Vespa,
    *,
    top_k: int = 3,
    save_dir: str = "retrieved_pages",
    resize: int | None = 640,
) -> List[str]:
    """
    Retrieve top_k CPIC pages for a query, save images, and return file paths.
    """
    # 1. Build query embedding
    device = next(model.parameters()).device
    batch = processor.process_queries([query])
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        emb = model(**batch)[0].cpu()
    query_embedding = {idx: vec.tolist() for idx, vec in enumerate(emb)}

    # 2. Vespa query
    response: VespaQueryResponse = vespa_app.query(
        yql=(
            "select name, path, image, page_number "
            "from pdf_page where userInput(@userQuery)"
        ),
        ranking="default",
        userQuery=query,
        hits=top_k,
        timeout=120,
        body={"input.query(qt)": query_embedding},
    )
    if not response.is_successful():
        raise RuntimeError(f"Vespa query failed: {response.get_error_message()}")

    # 3. Save page images
    os.makedirs(save_dir, exist_ok=True)
    saved: List[str] = []
    for rank, hit in enumerate(response.hits):
        pdf_path = hit["fields"]["path"]
        page_num = hit["fields"]["page_number"]
        title = Path(pdf_path).stem

        img = pdf_helper.open_pdf_page(pdf_path, page_num)
        if resize:
            img = pdf_helper.resize_image(img, resize)

        fname = f"{rank:02d}_{title}_p{page_num+1}.png"
        out = Path(save_dir) / fname
        img.save(out)
        saved.append(str(out))

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve CPIC pages via Vespa and save them"
    )
    parser.add_argument(
        "--query", required=True, help="Natural-language query"
    )
    parser.add_argument(
        "--endpoint", required=True,
        help="Vespa application endpoint URL"
    )
    parser.add_argument(
        "--model-name", default="vidore/colqwen2.5-v0.2"
    )
    parser.add_argument(
        "--cache-dir", default="/tmp/colqwen_cache"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="cuda:0 | mps | cpu"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of pages to retrieve"
    )
    parser.add_argument(
        "--save-dir", default="retrieved_pages",
        help="Directory to save images"
    )
    parser.add_argument(
        "--resize", type=int, default=640,
        help="Short side resize in pixels (None to skip)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load model
    device = args.device if torch.cuda.is_available() else "cpu"
    model, processor = load_model_and_processor(
        args.model_name, args.cache_dir, device
    )

    # Connect to Vespa
    app = Vespa(url=args.endpoint)

    # Retrieve
    paths = retrieve_cpic_pages(
        args.query,
        model,
        processor,
        app,
        top_k=args.top_k,
        save_dir=args.save_dir,
        resize=args.resize,
    )

    print("Saved images:")
    for p in paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
