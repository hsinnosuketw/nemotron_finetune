import json, sys, argparse
from query_rewrite.cpic_query_rewrite import query_rewrite
from prompt import system_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--question", required=True)
args = ap.parse_args()

queries = query_rewrite(
    question_prompt=args.question,
    system_prompt=system_prompt,
    num_tokens_to_generate=256,
    temperature=1.0,
    top_p=0.0,
    top_k=1,
)
print(json.dumps(queries, ensure_ascii=False))
