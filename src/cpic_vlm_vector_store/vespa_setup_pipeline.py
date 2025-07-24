#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPIC guideline end-to-end pipeline

Steps
-----
1. Read CPIC PDFs → extract images and texts.
2. Generate patch-level embeddings with ColQwen2-5.
3. Build a Vespa-compatible feed (JSON).
4. Optionally deploy schema + application to Vespa Cloud.
5. Feed pages into Vespa document store.

Execution
---------
python pipeline.py --cpic-dir /path/to/pdfs --deploy-vespa
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available
from vespa.application import Vespa
from vespa.deployment import VespaCloud
from vespa.io import VespaResponse
from vespa.package import (
    ApplicationPackage,
    Document,
    Field,
    FieldSet,
    FirstPhaseRanking,
    Function,
    HNSW,
    RankProfile,
    Schema,
    SecondPhaseRanking,
)

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import pdf_helper  # helper module

# Disable duplicate tokenizer workers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ----------------------------------------------------------------------------- #
#                               Model utilities                                 #
# ----------------------------------------------------------------------------- #

def load_model_and_processor(
    model_name: str,
    cache_dir: str,
    device: str,
) -> tuple[ColQwen2_5, ColQwen2_5_Processor]:
    """
    Load ColQwen2-5 model and its processor on the specified device.
    """
    model = ColQwen2_5.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else None
        ),
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(
        model_name, cache_dir=cache_dir)
    return model, processor


def embed_cpic_pdfs(
    cpic_pdfs: List[Dict],
    model: ColQwen2_5,
    processor: ColQwen2_5_Processor,
    batch_size: int,
    device: str,
) -> None:
    """
    In-place embedding generation for each PDF.
    Adds a new key "embeddings" to every item in cpic_pdfs.
    """
    for pdf in cpic_pdfs:
        page_embeddings = []
        dataloader = DataLoader(
            pdf["images"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=processor.process_images,
        )
        for batch in tqdm(dataloader, desc=f"Embedding {pdf['name']}"):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                emb = model(**batch)
                page_embeddings.extend(torch.unbind(emb.cpu()))
        pdf["embeddings"] = page_embeddings


# ----------------------------------------------------------------------------- #
#                           Vespa-related utilities                             #
# ----------------------------------------------------------------------------- #

def build_vespa_feed(cpic_pdfs: List[Dict]) -> List[Dict]:
    """
    Convert PDFs to Vespa feed format (list of dicts).
    """
    feed: List[Dict] = []
    for pdf in cpic_pdfs:
        for page_num, (txt, emb, img) in enumerate(
            zip(pdf["texts"], pdf["embeddings"], pdf["images"])
        ):
            emb_dict = {
                idx: (
                    np.packbits((patch > 0).cpu().numpy().astype(np.uint8))
                    .astype(np.int8)
                    .tobytes()
                    .hex()
                )
                for idx, patch in enumerate(emb)
            }
            feed.append(
                {
                    "id": pdf_helper.sha_id(pdf["name"], page_num),
                    "name": pdf["name"],
                    "path": pdf["path"],
                    "page_number": page_num,
                    "image": pdf_helper.image_to_base64(img),
                    "text": txt,
                    "embedding": emb_dict,
                }
            )
    return feed


def create_schema(schema_name: str = "pdf_page") -> Schema:
    """
    Return a Vespa schema with HNSW-backed embedding field.
    """
    document = Document(
        fields=[
            Field("id", "string", indexing=[
                  "summary", "index"], match=["word"]),
            Field("name", "string", indexing=["summary", "index"]),
            Field(
                "path",
                "string",
                indexing=["summary", "index"],
                match=["text"],
                index="enable-bm25",
            ),
            Field("page_number", "int", indexing=["summary", "attribute"]),
            Field("image", "raw", indexing=["summary"]),
            Field(
                "text",
                "string",
                indexing=["index"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(
                "embedding",
                "tensor<int8>(patch{}, v[16])",
                indexing=["attribute", "index"],
                ann=HNSW(
                    distance_metric="hamming",
                    max_links_per_node=8,
                    neighbors_to_explore_at_insert=100,
                ),
            ),
        ]
    )
    schema = Schema(
        name=schema_name,
        document=document,
        fieldsets=[FieldSet(name="default", fields=["name", "text"])],
    )
    profile = RankProfile(
        name="default",
        inputs=[("query(qt)", "tensor(querytoken{}, v[128])")],
        functions=[
            Function(
                "max_sim",
                (
                    "sum(reduce(sum(query(qt) * unpack_bits(attribute(embedding)), v), "
                    "max, patch), querytoken)"
                ),
            ),
            Function("bm25_score", "bm25(name) + bm25(text)"),
        ],
        first_phase=FirstPhaseRanking("bm25_score"),
        second_phase=SecondPhaseRanking("max_sim", rerank_count=100),
    )
    schema.add_rank_profile(profile)
    return schema


def deploy_to_vespa(
    schema: Schema,
    tenant_name: str,
    app_name: str,
    vespa_key: str | None = None,
) -> tuple[Vespa, str]:
    """
    Deploy the Vespa application and return (app_handle, public_endpoint_url).
    """
    package = ApplicationPackage(name=app_name, schema=[schema])
    vespa_cloud = VespaCloud(
        tenant=tenant_name,
        application=app_name,
        application_package=package,
        key_content=vespa_key,
    )

    app: Vespa = vespa_cloud.deploy()

    # --- get public (no-mTLS) URL from status JSON ---

    # persist status JSON (optional, useful for debugging)

    return app


def parse_args() -> argparse.Namespace:
    """
    Command-line interface for the pipeline.
    """
    parser = argparse.ArgumentParser(
        description="CPIC → Embeddings → Vespa pipeline"
    )
    parser.add_argument("--model-name", default="vidore/colqwen2.5-v0.2")
    parser.add_argument("--cache-dir", default="/tmp/colqwen_cache")
    parser.add_argument(
        "--device", default="cuda:0", help="cuda:0 | mps | cpu"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--cpic-dir", required=True, help="Directory with CPIC PDFs"
    )
    parser.add_argument(
        "--feed-output", default="vespa_feed.json"
    )
    parser.add_argument(
        "--deploy-vespa",
        action="store_true",
        help="Deploy schema and feed into Vespa",
    )
    parser.add_argument(
        "--tenant-name", default="mytenant"
    )
    parser.add_argument(
        "--vespa-app-name", default="cpicguidelinevrag"
    )
    parser.add_argument(
        "--vespa-key", default=None, help="Private key text"
    )
    return parser.parse_args()


async def feed_pages_to_vespa(
    vespa_app: Vespa,
    feed: List[Dict],
    schema: str = "pdf_page"
) -> None:
    """
    Asynchronously feed each document in 'feed' to Vespa under 'schema'.
    """
    async with vespa_app.asyncio(connections=1, timeout=180) as session:
        for page in tqdm(feed, desc="Feeding pages"):
            response: VespaResponse = await session.feed_data_point(
                data_id=page["id"],
                fields=page,
                schema=schema,
            )
            if not response.is_successful():
                print(f"Failed to feed {page['id']}: {response.json}")


def run(args: argparse.Namespace) -> None:
    """
    Pipeline driver.
    """
    # Step 1: Extract pages
    cpic_pdfs = pdf_helper.get_cpic_pdf_images_texts(args.cpic_dir)
    # Step 2: Load model
    device = args.device if torch.cuda.is_available() else "cpu"
    model, processor = load_model_and_processor(
        args.model_name, args.cache_dir, device
    )
    # Step 3: Embed pages
    embed_cpic_pdfs(
        cpic_pdfs, model, processor, args.batch_size, device
    )
    # Step 4: Build JSON feed
    feed = build_vespa_feed(cpic_pdfs)
    with open(args.feed_output, "w", encoding="utf-8") as fp:
        json.dump(feed, fp, ensure_ascii=False, indent=2)
    # Step 5: Optionally deploy & feed
    if args.deploy_vespa:
        schema = create_schema()
        app = deploy_to_vespa(
            schema,
            args.tenant_name,
            args.vespa_app_name,
            args.vespa_key,
        )

        # print & store for later use
        with open("vespa_endpoint.txt", "w", encoding="utf-8") as fh:
            fh.write("{\n")
            fh.write(f"\t'Endpoint':'{app.end_point}',\n")
            fh.write(f"\t'Secret Token':'{app.vespa_cloud_secret_token}',\n")
            fh.write(f"\t'Cert':'{app.cert}',\n")
            fh.write(f"\t'Key':'{app.key}',\n")
            fh.write(f"\t'URL':'{app.url}',\n")
            fh.write("\n}")

        # asyncio.run(feed_pages_to_vespa(app, feed))
    else:
        print("[i] Skipped Vespa deployment; feed JSON saved.")


if __name__ == "__main__":
    run(parse_args())
