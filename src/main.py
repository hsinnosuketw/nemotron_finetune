import json, subprocess, sys, pathlib
import os
import re

QUESTION = "dose adjustment for CYP2C19 poor metabolizer"
BASE = pathlib.Path(__file__).parent / "run_script"

# ------------------------------------------------------------
# (A) 直接用「目前執行 main.py」的解譯器跑 rewrite.py
# ------------------------------------------------------------
cmd = [
    "torchrun",
    "--standalone",
    "--nproc_per_node", "2",          # ← 依照 tensor model parallel size 調整
    str(BASE / "rewrite.py"),
    "--question", QUESTION,
]

# (可選) 避免某些 NCCL 啟動警告
env = os.environ.copy()
env.setdefault("NCCL_LAUNCH_MODE", "PARALLEL")


raw = subprocess.check_output(cmd, text=True, env=env)
# qlist = json.loads(result)

match = re.search(r"Assistant\s*(.*?)\s*<extra_id", raw, re.S)
rewritten_query = match.group(1).strip() if match else None

print("Rewritten: ", rewritten_query)

# ------------------------------------------------------------
# (B) 在 conda 環境 `vespa_env` 內跑 vespa_query.py
#     → 使用 conda run -n <env> python ...
# ------------------------------------------------------------
subprocess.check_output(
    [
        "conda", "run", "-n", "vespa_env", "python",
        BASE / "retrieve_cpic.py",
        "--query", rewritten_query,
        "--endpoint-file", "vespa_endpoint.txt",
        "--top-k", "3",
    ],
    text=True,
)


# ------------------------------------------------------------
# (C) extract_page.py 同樣在 vespa_env 執行
# ------------------------------------------------------------

RETRIEVED_DIR = pathlib.Path("./retrieved_pages")
PROMPT_TXT    = "Extract table from this page (markdown)."

for cpic_single_page_png_path in RETRIEVED_DIR.glob("*.png"):
    print(f"→ processing: {cpic_single_page_png_path.name}")

    subprocess.run(
        [
            "conda", "run", "-n", "vespa_env", "python",
            BASE / "vlm_extract.py",
            "--pdf", str(cpic_single_page_png_path),
            "--page-idx", "0",              # 一律第 1 頁；可自行帶參數
            "--prompt", PROMPT_TXT,
        ],
        check=True,
    )