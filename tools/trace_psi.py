# -*- coding: utf-8 -*-
"""逐行追踪 PSI 的计算过程（只看第 1 个特征）。"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)

expected = np.load(r"D:\人工智能+\data\gas\batch_01.npz")["X"][:, 0]
actual = np.load(r"D:\人工智能+\data\gas\batch_04.npz")["X"][:, 0]
bins, eps = 10, 1e-6

print("=" * 78)
print("输入")
print("=" * 78)
print(f"  expected（参照系 batch_01）= {expected.size} 个读数")
print(f"  actual  （当前   batch_04）= {actual.size} 个读数")

print("\n" + "=" * 78)
print("第 1 行：qs = np.linspace(0, 1, bins + 1)")
print("=" * 78)
qs = np.linspace(0, 1, bins + 1)
print(f"  {qs}")

print("\n" + "=" * 78)
print("第 2 行：edges = np.unique(np.quantile(expected, qs))")
print("=" * 78)
raw = np.quantile(expected, qs)
print("  参照系按这些分位点取出的数值：")
for i, (q, v) in enumerate(zip(qs, raw)):
    print(f"    {q:>4.1f} 分位 → {v:>12.1f}")
edges = np.unique(raw)

print("\n" + "=" * 78)
print("第 3~4 行：两头放开")
print("=" * 78)
edges[0] = -np.inf
edges[-1] = np.inf
print(f"  edges = [{edges[0]}, {edges[1]:.1f}, {edges[2]:.1f}, ..., {edges[-2]:.1f}, {edges[-1]}]")
print(f"  共 {edges.size} 条边界 → {edges.size - 1} 个箱子")

print("\n" + "=" * 78)
print("第 6~7 行：数每个箱子里有多少个，再除以总数")
print("=" * 78)
e_cnt = np.histogram(expected, bins=edges)[0]
a_cnt = np.histogram(actual, bins=edges)[0]
e = e_cnt / expected.size
a = a_cnt / actual.size
print(f"{'箱':<4}{'参照系个数':>10}{'占比 e':>9}{'当前个数':>10}{'占比 a':>9}")
print("-" * 44)
for i in range(len(e)):
    print(f"{i+1:<4}{e_cnt[i]:>10}{e[i]:>9.3f}{a_cnt[i]:>10}{a[i]:>9.3f}")
print("-" * 44)
print(f"{'合计':<4}{e_cnt.sum():>10}{e.sum():>9.3f}{a_cnt.sum():>10}{a.sum():>9.3f}")

print("\n" + "=" * 78)
print("第 9~10 行：np.clip(e, eps) —— 防止某箱是 0 导致 ln(0)")
print("=" * 78)
e_c = np.clip(e, eps, None)
a_c = np.clip(a, eps, None)
changed = int((e_c != e).sum() + (a_c != a).sum())
print(f"  这一批有没有箱子是 0 个？ {'有，被抬到 1e-6' if changed else '没有，不用动'}")

print("\n" + "=" * 78)
print("第 12 行：sum((a - e) * log(a / e))")
print("=" * 78)
print(f"{'箱':<4}{'a-e':>9}{'a/e':>9}{'ln(a/e)':>10}{'乘积':>10}")
print("-" * 44)
total = 0.0
for i in range(len(e)):
    diff = a_c[i] - e_c[i]
    ratio = a_c[i] / e_c[i]
    ln = np.log(ratio)
    term = diff * ln
    total += term
    print(f"{i+1:<4}{diff:>+9.3f}{ratio:>9.3f}{ln:>+10.3f}{term:>+10.4f}")
print("-" * 44)
print(f"{'PSI =':<4}{'':>9}{'':>9}{'':>10}{total:>+10.4f}")

print("\n" + "=" * 78)
print("验证：直接调用代码里的 _psi()")
print("=" * 78)
import sys
sys.path.insert(0, r"D:\人工智能+")
from evolver.loop.monitor import _psi
print(f"  _psi(expected, actual, bins=10) = {_psi(expected, actual, bins):.4f}")
print(f"  手算结果                         = {total:.4f}")
