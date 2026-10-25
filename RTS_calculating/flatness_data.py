#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯平滑数据.py (保形高斯平滑版)

核心原则：
  1. 不使用 SG 滤波和插值，避免多项式拟合导致的假峰/震荡。
  2. 自动检测数据断层，分段平滑，绝不跨越空白区域连线。
  3. 使用高斯低通滤波，纯粹压制高频毛刺，完美保留原始包络形状。

输出:
    exp_data.txt / sif_data.txt          —— 两列 x y，纯数值
    compare_exp.png / compare_sif.png    —— 原始 vs 平滑对比图
"""

import os
import sys
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==================== 配置 ====================
INPUT_FILES  = {"green": "exp_0.txt", "blue": "sif_0.txt"}
OUTPUT_FILES = {"green": "exp_data.txt", "blue": "sif_data.txt"}
PLOT_FILES   = {"green": "compare_exp.png", "blue": "compare_sif.png"}

# 【核心调参区】高斯平滑的强度 (sigma)。
# 越小越保形但去噪弱，越大越平滑但可能削平极尖的峰。
# 结合你的图，建议先试 2.0 左右，若不够平滑再加到 4.0。
PARAMS = {
    "green": {"sigma": 2.5},   # 绿线整体噪声大，可以稍微大一点
    "blue":  {"sigma": 1.5},   # 蓝线要保护峰顶，用较小的 sigma
}

# 断层检测阈值：如果相邻点的 x 间距 > 5倍的中位间距，则认为是断层，不跨断层平滑
GAP_THRESHOLD = 5.0

ZOOM_FRAC = 0.25
POINT_ALPHA = 0.25
POINT_SIZE = 1.5
LINE_WIDTH = 1.6
# ==============================================


def load_xy(path):
    xs, ys = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except ValueError:
                continue
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x); ys.append(y)
    return np.asarray(xs, float), np.asarray(ys, float)


def unify_x(x, y):
    """同一 x 下多个 y 取中值，去重并排序"""
    order = np.argsort(x, kind="mergesort")
    x = x[order]; y = y[order]
    xs, ys = [], []
    i, n = 0, len(x)
    while i < n:
        j = i + 1
        while j < n and x[j] == x[i]:
            j += 1
        xs.append(float(x[i]))
        ys.append(float(np.median(y[i:j])))
        i = j
    return np.asarray(xs), np.asarray(ys)


def denoise(x, y, sigma):
    """
    严格保形平滑：按 x 排序 -> 找断层 -> 分段插值成均匀网格 -> 高斯平滑。
    """
    xu, yu = unify_x(x, y)
    if len(xu) < 3:
        return xu, yu

    # 估算正常点间距
    dxs = np.diff(xu)
    dxs = dxs[dxs > 0]
    if len(dxs) == 0:
        return xu, yu
    median_dx = np.median(dxs)

    # 找断层索引
    gap_indices = np.where(np.diff(xu) > GAP_THRESHOLD * median_dx)[0] + 1
    segments = np.split(np.arange(len(xu)), gap_indices)

    out_x, out_y = [], []

    for seg_idx in segments:
        if len(seg_idx) < 3:
            out_x.extend(xu[seg_idx])
            out_y.extend(yu[seg_idx])
            continue

        x_seg = xu[seg_idx]
        y_seg = yu[seg_idx]

        # 仅在段内生成均匀网格（步长用整体中位间距）
        grid = np.arange(x_seg[0], x_seg[-1], median_dx)
        if grid[-1] < x_seg[-1]:
            grid = np.append(grid, x_seg[-1])

        # 段内线性插值
        y_interp = np.interp(grid, x_seg, y_seg)

        # 高斯滤波（纯加权平均，绝对不会产生假峰）
        y_smooth = gaussian_filter1d(y_interp, sigma=sigma, mode="nearest")

        out_x.extend(grid)
        out_y.extend(y_smooth)

    return np.asarray(out_x), np.asarray(out_y)


def write_two_cols(path, x, y):
    ap = os.path.abspath(path)
    with open(ap, "w", encoding="utf-8") as f:
        for xi, yi in zip(x, y):
            f.write(f"{xi:.12g}\t{yi:.12g}\n")
    print(f"  data -> {ap}  ({len(x)} rows)")


def make_compare_plot(name, x_raw, y_raw, x_sm, y_sm, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1.4]})

    # 上图：全局对比
    ax = axes[0]
    ax.scatter(x_raw, y_raw, s=POINT_SIZE, alpha=POINT_ALPHA, color="tab:gray", label="raw")
    ax.plot(x_sm, y_sm, "-", lw=LINE_WIDTH, color="tab:red", label="denoised")
    ax.set_title(f"{name}: raw vs denoised (Gaussian Filter)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # 计算放大区域
    xmin, xmax = np.min(x_sm), np.max(x_sm)
    span = xmax - xmin if xmax > xmin else 1.0
    zc = 0.5 * (xmin + xmax)
    half = 0.5 * ZOOM_FRAC * span
    zlo, zhi = zc - half, zc + half

    for xv in (zlo, zhi):
        ax.axvline(xv, color="tab:blue", ls="--", lw=0.8, alpha=0.6)

    # 下图：局部放大
    ax2 = axes[1]
    m_raw = (x_raw >= zlo) & (x_raw <= zhi)
    m_sm = (x_sm >= zlo) & (x_sm <= zhi)
    ax2.scatter(x_raw[m_raw], y_raw[m_raw], s=POINT_SIZE * 2, alpha=0.5, color="tab:gray")
    ax2.plot(x_sm[m_sm], y_sm[m_sm], "-", lw=LINE_WIDTH, color="tab:red")
    ax2.set_xlim(zlo, zhi)
    ax2.set_title(f"zoom: [{zlo:.4g}, {zhi:.4g}]")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    ap = os.path.abspath(out_path)
    fig.savefig(ap, dpi=180)
    plt.close(fig)
    print(f"  plot -> {ap}")


def main():
    print("=" * 50)
    for name in ("green", "blue"):
        print(f"\n--- {name} ---")
        inp = os.path.abspath(INPUT_FILES[name])
        if not os.path.exists(inp):
            print(f"  [SKIP] 文件不存在: {inp}")
            continue

        try:
            x, y = load_xy(inp)
            if len(x) < 3:
                print("  点太少，跳过")
                continue

            sigma = PARAMS[name]["sigma"]
            xo, yo = denoise(x, y, sigma)

            write_two_cols(OUTPUT_FILES[name], xo, yo)
            make_compare_plot(name, x, y, xo, yo, PLOT_FILES[name])

        except Exception:
            print(f"  [ERROR] 处理 {name} 失败:")
            traceback.print_exc()
    print("\n处理完成。")


if __name__ == "__main__":
    main()