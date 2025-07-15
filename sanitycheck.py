import json, sys, unicodedata, pathlib

bad = []
path = pathlib.Path(sys.argv[1])
for ln, line in enumerate(path.open(encoding="utf-8"), 1):
    try:
        obj = json.loads(line)
    except Exception as e:
        bad.append((ln, f"json error: {e}"))
        continue
    # 檢查 printable
    raw = line.rstrip("\n")
    if any(ch in ("\u000b", "\u000c", "\u2028", "\u2029") for ch in raw):
        bad.append((ln, "non-printable char"))
    # 檢查最長長度（視自己 GPU 調）
    if len(raw) > 32000:
        bad.append((ln, f"line too long: {len(raw)}"))

if bad:
    for ln, msg in bad[:10]:
        print(f"行 {ln}: {msg}")
    print(f"✗ 共 {len(bad)} 行有問題")
else:
    print("✓ deep scan ok")
