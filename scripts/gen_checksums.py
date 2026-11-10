# -*- coding: utf-8 -*-
"""生成 / 校验 checksums.sha256。

     python scripts/gen_checksums.py --write    # 生成或更新清单
     python scripts/gen_checksums.py --verify   # 校验（默认）

为什么需要：数据版本一变，所有实验结论都不可复现。
这份清单记录"当前跑出结果的到底是哪份数据"。

注意：data/ 已被 .gitignore 忽略，大数据不进仓库；
checksums.sha256 本身很小，应当提交，用来约束"队友本地那份数据是否一致"。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "checksums.sha256")

# 只收录 data/<dataset>/{raw,processed} 下的数据文件；
# 额外把 data/<dataset>/ 根部也扫一遍，防止有人把 npz 直接放在根目录而漏掉哈希
DIR_GLOBS = [
    os.path.join(REPO, "data", "*", "raw"),
    os.path.join(REPO, "data", "*", "processed"),
    os.path.join(REPO, "data", "*"),
]
EXTS = {".npz", ".zip", ".dat", ".txt", ".csv", ".gz", ".7z", ".rar"}


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def collect() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern in DIR_GLOBS:
        for base in glob.glob(pattern):
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                p = os.path.join(base, name)
                if not os.path.isfile(p):
                    continue
                if os.path.splitext(name)[1].lower() not in EXTS:
                    continue
                rel = os.path.relpath(p, REPO).replace(os.sep, "/")
                if rel in seen:
                    continue
                seen.add(rel)
                items.append((sha256_of(p), rel))
    items.sort(key=lambda x: x[1])
    return items


def write() -> int:
    items = collect()
    if not items:
        print("[!] 没有找到任何数据文件；请先把数据放到 data/<dataset>/{raw,processed}/")
        return 1
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        for digest, rel in items:
            f.write(f"{digest}  {rel}\n")
    print(f"[ok] 写入 {os.path.relpath(MANIFEST, REPO)}：{len(items)} 个文件")
    for digest, rel in items:
        print(f"     {digest[:16]}…  {rel}")
    return 0


def verify() -> int:
    if not os.path.exists(MANIFEST):
        print("[!] checksums.sha256 不存在，先跑 --write")
        return 1
    expected = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, rel = line.split(None, 1)
            expected[rel.strip()] = digest

    bad, missing, ok = [], [], 0
    for rel, digest in expected.items():
        p = os.path.join(REPO, rel.replace("/", os.sep))
        if not os.path.exists(p):
            missing.append(rel)
            continue
        actual = sha256_of(p)
        if actual == digest:
            ok += 1
        else:
            bad.append((rel, digest, actual))

    print(f"校验：{ok} 通过 / {len(bad)} 不一致 / {len(missing)} 缺失")
    for rel, e, a in bad:
        print(f"  [X] {rel}\n      期望 {e[:16]}…  实际 {a[:16]}…")
    for rel in missing:
        print(f"  [-] 缺失 {rel}")
    return 0 if (not bad and not missing) else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="生成/更新清单")
    ap.add_argument("--verify", action="store_true", help="校验（默认）")
    a = ap.parse_args()
    raise SystemExit(write() if a.write else verify())
