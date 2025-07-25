import json, subprocess, sys, pathlib

QUESTION = "dose adjustment for CYP2C19 poor metabolizer"
BASE = pathlib.Path(__file__).parent / "run_script"

# ------------------------------------------------------------
# (A) 直接用「目前執行 main.py」的解譯器跑 rewrite.py
# ------------------------------------------------------------
result = subprocess.check_output(
    [
        sys.executable,                 # ← 當前 Python (無虛擬環境需求)
        BASE / "rewrite.py",
        "--question", QUESTION,
    ],
    text=True,
)
qlist = json.loads(result)
print("Rewritten:", qlist)

# ------------------------------------------------------------
# (B) 在 conda 環境 `vespa_env` 內跑 vespa_query.py
#     → 使用 conda run -n <env> python ...
# ------------------------------------------------------------
result = subprocess.check_output(
    [
        "conda", "run", "-n", "vespa_env", "python",
        BASE / "vespa_query.py",
        "--query", QUESTION,
        "--endpoint-file", "vespa_endpoint.txt",
        "--topk", "3",
    ],
    text=True,
)
files = json.loads(result)
print("Files:", files)

# ------------------------------------------------------------
# (C) extract_page.py 同樣在 vespa_env 執行
# ------------------------------------------------------------
subprocess.run(
    [
        "conda", "run", "-n", "vespa_env", "python",
        BASE / "vlm_extract.py",
        "--pdf", files[0],    # 第一名頁面
        "--page-idx", "0",
        "--prompt", "Extract table ...",
    ],
    check=True,
)
