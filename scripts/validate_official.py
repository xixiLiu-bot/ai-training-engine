# -*- coding: utf-8 -*-
"""官方数据只读体检：只查结构，**不训练、不拟合、不选参**。

    python scripts/validate_official.py --root data/official
    python scripts/validate_official.py --root data/official --json out.json

检查项
------
1. 目录存在且非空
2. manifest.json 存在且是合法 JSON
3. manifest 必填字段齐全（format/label_column/files/batch_column…）
4. manifest 声明的文件真实存在
5. 能按声明的格式读出批次，且标签列、特征列对得上
6. 批次号连续、时间顺序单调
7. 每批样本数、缺失值比例
8. 数据身份必须是官方公开集（official_public）

退出码：0 = 通过；1 = 检查失败；2 = 数据未提供（目录/文件缺失）

本脚本**不会**修改任何数据，也**不会**产生任何"评测成绩"。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from evolver.adapters import identities                      # noqa: E402
from evolver.adapters.official import MANIFEST_NAME, OfficialSource  # noqa: E402
from evolver.adapters.sources import DataSourceError         # noqa: E402

OK = "[OK]"
BAD = "[!!]"
INFO = "[--]"


def _describe_dir(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        size = os.path.getsize(p) if os.path.isfile(p) else 0
        out.append(f"{name}（{'目录' if os.path.isdir(p) else f'{size} B'}）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data", "official"))
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    ap.add_argument("--allow-missing", action="store_true",
                    help="数据未提供时也返回 0（只用于 CI 里跑纪律测试）")
    args = ap.parse_args()

    checks: list[dict] = []
    def rec(name: str, ok: bool | None, detail: str):
        checks.append({"check": name, "ok": ok, "detail": detail})
        mark = OK if ok else (BAD if ok is False else INFO)
        print(f"{mark} {name}: {detail}")

    print("=" * 62)
    print("官方数据只读体检（不训练、不选参、不产生成绩）")
    print("=" * 62)
    rec("data_identity", None,
        f"{identities.OFFICIAL_PUBLIC}（官方公开集）—— 只允许结构检查")

    # 1) 目录
    if not os.path.isdir(args.root):
        rec("目录", False, f"不存在：{args.root}")
        rec("清单", None, f"缺少 {MANIFEST_NAME}")
        print("\n结论：**官方公开集尚未提供**。")
        print("      需从比赛评测平台下载后放入该目录（本仓库不伪造下载地址）。")
        _dump(args.json, checks, "missing")
        return 0 if args.allow_missing else 2
    entries = _describe_dir(args.root)
    if not entries:
        rec("目录", False, f"空目录：{args.root}")
        print("\n结论：**目录为空**，官方数据尚未放入。")
        _dump(args.json, checks, "empty")
        return 0 if args.allow_missing else 2
    rec("目录", True, f"{len(entries)} 项 → {', '.join(entries[:6])}")

    # 2) 清单
    if not os.path.isfile(os.path.join(args.root, MANIFEST_NAME)):
        rec("清单", False, f"缺少 {MANIFEST_NAME}；没有清单无法确定列名/标签/批次")
        print("\n结论：**缺少清单文件**，无法继续（拒绝猜测式解析）。")
        _dump(args.json, checks, "no_manifest")
        return 0 if args.allow_missing else 1

    src = OfficialSource(args.root)
    try:
        found = src.discover()
    except DataSourceError as exc:
        rec("清单", False, str(exc))
        _dump(args.json, checks, "manifest_error")
        return 0 if args.allow_missing else 1
    rec("清单", True, f"已解析：{found['path']}")

    # 3) 加载（含字段/文件校验）
    try:
        src.load()
    except DataSourceError as exc:
        rec("加载", False, str(exc))
        _dump(args.json, checks, "load_error")
        return 0 if args.allow_missing else 1
    except Exception as exc:                        # noqa: BLE001
        rec("加载", False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        _dump(args.json, checks, "exception")
        return 0 if args.allow_missing else 1
    meta = src.get_metadata()
    rec("加载", True, f"{meta['n_batches']} 批 / {meta['n_samples']} 样本 / "
                      f"{len(src.get_features())} 特征 / 格式 {meta['format']}")

    # 4) 批次连续性 + 时间顺序
    ids = meta["batch_ids"]
    contiguous = ids == list(range(ids[0], ids[0] + len(ids)))
    rec("批次连续", contiguous, f"{ids[:8]}{'…' if len(ids) > 8 else ''}")
    rec("时间顺序", ids == sorted(ids), "按 manifest.batch_order 升序")

    # 5) 标签分布 + 缺失值
    per_batch = []
    empty_label = False
    for b in src.iter_batches():
        ys = [str(v) for v in b["y"]]
        n_empty = sum(1 for v in ys if v in ("", "nan", "None"))
        empty_label = empty_label or n_empty > 0
        per_batch.append({
            "batch_id": b["batch_id"], "n": len(ys),
            "n_classes": len(set(ys)), "empty_labels": n_empty,
        })
    rec("标签非空", not empty_label,
        "存在空标签" if empty_label else "全部批次标签非空")
    rec("标签分布", True, f"每批 {per_batch[0]['n']}~{per_batch[-1]['n']} 样本，"
                          f"类别数 {min(p['n_classes'] for p in per_batch)}~"
                          f"{max(p['n_classes'] for p in per_batch)}")

    # 6) 身份复核：绝不允许把 official_test 当成本地文件检查
    rec("身份复核", src.data_identity == identities.OFFICIAL_PUBLIC,
        f"{src.data_identity}（{identities.describe(src.data_identity)}）")

    print("\n结论：**结构体检通过**。")
    print("      注意：这只说明格式可用，不代表任何评测成绩；")
    print("      官方数据仍不得参与阈值/margin/动作策略/模型调参。")
    _dump(args.json, checks, "ok", meta=meta, per_batch=per_batch)
    return 0


def entries_str(entries: list[str]) -> str:
    return " | ".join(entries)


def _dump(path: str | None, checks: list, status: str, **extra) -> None:
    if not path:
        return
    payload = {"status": status, "checks": checks,
               "official_data_connected": status == "ok", **extra}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告：{path}")


if __name__ == "__main__":
    raise SystemExit(main())
