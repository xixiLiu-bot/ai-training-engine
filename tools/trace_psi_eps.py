# -*- coding: utf-8 -*-
"""为什么 eps 一改，PSI 就差这么多？—— 用气体数据逐项拆开。"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)

E = np.load(r"D:\人工智能+\data\gas\batch_01.npz")["X"][:, 0]
A = np.load(r"D:\人工智能+\data\gas\batch_04.npz")["X"][:, 0]

edges = np.unique(np.quantile(E, np.linspace(0, 1, 11)))
edges[0], edges[-1] = -np.inf, np.inf
e = np.histogram(E, bins=edges)[0] / len(E)
a = np.histogram(A, bins=edges)[0] / len(A)

print("=" * 78)
print("事实：10 个箱子里，有 3 个在『当前批』里一个样本都没有")
print("=" * 78)
empty = np.where(a == 0)[0]
for i in empty:
    print(f"  第 {i+1} 箱：参照系占比 {e[i]:.3f}，当前批 0 个样本 → 占比 0")

print("\n" + "=" * 78)
print("问题：公式里有 ln(a/e)，a=0 时 ln(0) = 负无穷，算不了")
print("=" * 78)
print("  所以代码用 np.clip(a, eps) 把 0 换成一个极小的数，比如 1e-6")
print("  但这个『极小』换成多少，会直接决定结果——因为它在 log 里面")

print("\n" + "=" * 78)
print("看第 4 箱（e=0.099, a=0）在不同 eps 下的贡献")
print("=" * 78)
print(f"{'eps':>10}{'a 被换成':>12}{'a/e':>12}{'ln(a/e)':>11}{'贡献':>12}")
print("-" * 60)
e4 = e[3]
for eps in (1e-2, 1e-4, 1e-6, 1e-8, 1e-12):
    ratio = eps / e4
    ln = np.log(ratio)
    term = (eps - e4) * ln
    print(f"{eps:>10.0e}{eps:>12.2e}{ratio:>12.2e}{ln:>+11.2f}{term:>+12.3f}")

print("\n  ↑ 注意最后一列：同一个空箱，贡献从 +0.46 一路涨到 +2.51")
print("    因为 eps 每小 100 倍，ln 就再负 4.6，贡献就再加 0.46")

print("\n" + "=" * 78)
print("全量对比：eps 对总 PSI 的影响")
print("=" * 78)
print(f"{'eps':>10}{'非空箱贡献':>13}{'3 个空箱贡献':>15}{'合计 PSI':>12}")
print("-" * 54)
for eps in (1e-2, 1e-4, 1e-6, 1e-8, 1e-12):
    ec, ac = np.clip(e, eps, None), np.clip(a, eps, None)
    terms = (ac - ec) * np.log(ac / ec)
    non_empty = terms[a > 0].sum()
    emp = terms[a == 0].sum()
    print(f"{eps:>10.0e}{non_empty:>13.3f}{emp:>15.3f}{non_empty + emp:>12.3f}")

print("\n  ↑ 非空箱的贡献几乎不变（都是 0.57 左右）")
print("    全部差异都来自那 3 个空箱")

print("\n" + "=" * 78)
print("正确做法：空箱不『假装有个极小值』，而是合并到相邻箱")
print("=" * 78)


def merge_empty(e_cnt, a_cnt):
    """把当前批为 0 的箱子，与右侧（或左侧）箱子合并，直到没有空箱。"""
    e_cnt = list(e_cnt)
    a_cnt = list(a_cnt)
    changed = True
    while changed and len(e_cnt) > 1:
        changed = False
        for i in range(len(a_cnt)):
            if a_cnt[i] == 0:
                j = i + 1 if i + 1 < len(a_cnt) else i - 1
                lo, hi = min(i, j), max(i, j)
                e_cnt[lo] += e_cnt[hi]
                a_cnt[lo] += a_cnt[hi]
                del e_cnt[hi]
                del a_cnt[hi]
                changed = True
                break
    return np.array(e_cnt), np.array(a_cnt)


e_cnt_raw = np.histogram(E, bins=edges)[0].astype(float)
a_cnt_raw = np.histogram(A, bins=edges)[0].astype(float)
ec_m, ac_m = merge_empty(e_cnt_raw, a_cnt_raw)
e_m, a_m = ec_m / ec_m.sum(), ac_m / ac_m.sum()

print(f"  合并前：{len(e_cnt_raw)} 个箱子，3 个是空的")
print(f"  合并后：{len(ec_m)} 个箱子，0 个是空的")
print()
print(f"{'箱':<4}{'参照系占比':>11}{'当前占比':>10}")
print("-" * 26)
for i in range(len(e_m)):
    print(f"{i+1:<4}{e_m[i]:>11.3f}{a_m[i]:>10.3f}")
psi_merged = float(np.sum((a_m - e_m) * np.log(a_m / e_m)))
print("-" * 26)
print(f"  合并后的 PSI = {psi_merged:.4f}")
print("  → 没有空箱了，所以结果和 eps 完全无关（换多少 eps 都一样）")
print()
print("  对比：")
print(f"    原始 PSI (eps=1e-6)  = {np.sum((np.clip(a,1e-6,None)-np.clip(e,1e-6,None))*np.log(np.clip(a,1e-6,None)/np.clip(e,1e-6,None))):.4f}")
print(f"    合并空箱后           = {psi_merged:.4f}")
