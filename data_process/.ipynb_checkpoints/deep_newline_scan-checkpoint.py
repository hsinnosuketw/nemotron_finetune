# deep_newline_scan.py
import pathlib, sys, unicodedata, json

bad = []
path = pathlib.Path(sys.argv[1])

with path.open("r", encoding="utf-8", newline="") as f:  # newline="" => 完全不要轉譯
    for ln, raw in enumerate(f, 1):
        # 去掉真正的行尾（最後 1~2 個 \r\n 或 \n）
        if raw.endswith("\r\n"):
            raw_core = raw[:-2]
        elif raw.endswith("\n"):
            raw_core = raw[:-1]
        else:
            raw_core = raw
        for bad_ch in ("\n", "\r", "\u2028", "\u2029"):
            if bad_ch in raw_core:
                bad.append((ln, unicodedata.name(bad_ch)))
                break

if not bad:
    print("✓ 沒找到殘留換行符")
else:
    print(f"✗ 共 {len(bad)} 行含隱藏換行，前 10 行：")
    for ln, name in bad[:10]:
        print(f"  行 {ln}:  {name}")
