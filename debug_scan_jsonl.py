# debug_scan_jsonl.py
import json, sys, pathlib

def scan(path):
    bad = []
    for ln, line in enumerate(path.open(encoding="utf-8"), 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            bad.append((ln, e, line[:120]))
            if len(bad) >= 10:          # 只印前 10 行
                break
    return bad

if __name__ == "__main__":
    p = pathlib.Path(sys.argv[1])
    errs = scan(p)
    if not errs:
        print("✓ 檔案無格式錯誤")
    else:
        print(f"✗ 共 {len(errs)} 行失敗，以下列出前 {len(errs)} 行：")
        for ln, err, snippet in errs:
            print(f"  行 {ln:>6}: {err}｜{snippet}")
