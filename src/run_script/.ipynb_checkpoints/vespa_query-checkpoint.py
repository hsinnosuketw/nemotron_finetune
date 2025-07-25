import json, argparse, pathlib
from cpic_vlm_vector_store.cpic_pagewise_query import cpic_query

ap = argparse.ArgumentParser()
ap.add_argument("--query", required=True)
ap.add_argument("--endpoint-file", required=True)
ap.add_argument("--topk", type=int, default=3)
args = ap.parse_args()

files = cpic_query(
    query=args.query,
    endpoint_file=args.endpoint_file,
    top_k=args.topk,
)
print(json.dumps(files, ensure_ascii=False))
