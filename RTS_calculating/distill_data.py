#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_raw.py
只从图片提取蓝/绿谱线像素对应的 (x, y) 数值，不做任何平滑/追踪/过滤。

绿色 -> exp_0.txt
蓝色 -> sif_0.txt

输出：纯两列数值，制表符分隔，无表头、无注释。
同时生成对比图：左图原图+红线描点，右图提取出的(x,y)数据。

依赖:
    pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================== 用户配置 ==================
IMAGE_PATH = "figure.png"
USE_INTERACTIVE = False

MANUAL_CALIB = {
    "x_px":  [508.0, 808.0],
    "x_val": [0.0, -40.0],
    "y_px":  [788.0, 285.0],       # 图片像素行号，向下增大
    "y_val": [0.0, 2.0],
}

X_LOG = False
Y_LOG = False

# HSV 颜色范围 (OpenCV: H∈[0,180], S,V∈[0,255])
COLOR_RANGES = {
    "blue":  ([88, 50, 30], [145, 255, 255]),
    "green": ([30, 35, 30], [95, 255, 255]),
}

OUTPUT_FILES = {
    "green": "exp_0.txt",
    "blue":  "sif_0.txt",
}

COMPARE_FILES = {
    "green": "compare_extract_exp.png",
    "blue":  "compare_extract_sif.png",
}
# =============================================


def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def pick_two_points(img_bgr, title):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title(title)
    pts = plt.ginput(2, timeout=0, show_clicks=True)
    plt.close(fig)
    if len(pts) < 2:
        raise RuntimeError("没有选到两个点")
    return np.array(pts, dtype=float)


def get_calibration(img_bgr):
    if USE_INTERACTIVE:
        xp = pick_two_points(img_bgr, "Step 1: click two X-axis ticks")
        print("X pixel cols:", xp[:, 0])
        xv = [float(v) for v in input("X values (comma): ").split(",")]
        yp = pick_two_points(img_bgr, "Step 2: click two Y-axis ticks")
        print("Y pixel rows:", yp[:, 1])
        yv = [float(v) for v in input("Y values (comma): ").split(",")]
        x_px, y_px = xp[:, 0], yp[:, 1]
    else:
        x_px = np.asarray(MANUAL_CALIB["x_px"], float)
        xv = np.asarray(MANUAL_CALIB["x_val"], float)
        y_px = np.asarray(MANUAL_CALIB["y_px"], float)
        yv = np.asarray(MANUAL_CALIB["y_val"], float)

    fx = np.log10 if X_LOG else (lambda v: v)
    fy = np.log10 if Y_LOG else (lambda v: v)
    a_x, b_x = np.polyfit(x_px, fx(np.asarray(xv, float)), 1)
    a_y, b_y = np.polyfit(y_px, fy(np.asarray(yv, float)), 1)
    print(f"X mapping: {'log10(x)' if X_LOG else 'x'} = {a_x:.12g}*col + {b_x:.12g}")
    print(f"Y mapping: {'log10(y)' if Y_LOG else 'y'} = {a_y:.12g}*row + {b_y:.12g}")
    return a_x, b_x, a_y, b_y


def color_mask(img_bgr, lower, upper):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))


def to_x(a_x, b_x, col):
    v = a_x * col + b_x
    return 10.0 ** v if X_LOG else v


def to_y(a_y, b_y, row):
    v = a_y * row + b_y
    return 10.0 ** v if Y_LOG else v


def extract_segments(mask, a_x, b_x, a_y, b_y):
    """每列按连续像素段拆分；每段输出一个 (x, y)，y 为段内像素行均值。"""
    out = []
    for col in range(mask.shape[1]):
        rows = np.where(mask[:, col] > 0)[0]
        if rows.size == 0:
            continue
        splits = np.where(np.diff(rows) > 1)[0] + 1
        for seg in np.split(rows, splits):
            if seg.size == 0:
                continue
            x_d = to_x(a_x, b_x, col)
            y_d = to_y(a_y, b_y, float(seg.mean()))
            out.append((x_d, y_d, col, float(seg.mean())))
    return out


def write_two_cols(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for x_d, y_d, _c, _r in rows:
            f.write(f"{x_d:.12g}\t{y_d:.12g}\n")
    print(f"wrote {len(rows)} rows -> {path}")


def save_comparison(img_bgr, rows, path, name):
    """
    生成并排对比图：
    左图：原图 + 红色描点（展示提取位置）
    右图：提取出的 (x, y) 数据（展示物理数值）
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- 1. 左图：原图与描红 ---
    ax1 = axes[0]
    ax1.imshow(img_rgb)
    if rows:
        cols = np.array([r[2] for r in rows], float)
        rows_px = np.array([r[3] for r in rows], float)
        ax1.plot(cols, rows_px, '.', ms=1.5, color='red', label=f"Extracted ({len(rows)} pts)")
        ax1.legend(loc="upper right")
    ax1.set_title(f"Original Image & Overlay ({name})")
    ax1.axis('off') # 隐藏坐标轴边框，看起来更干净
    
    # --- 2. 右图：提取后的数据 ---
    ax2 = axes[1]
    if rows:
        x_data = np.array([r[0] for r in rows], float)
        y_data = np.array([r[1] for r in rows], float)
        # 用极小的点画出提取结果
        ax2.plot(x_data, y_data, '.', ms=2, color='blue', alpha=0.7)
    ax2.set_title(f"Extracted Data: {name}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[{name}] comparison -> {path}")


def main():
    img_bgr = load_image(IMAGE_PATH)
    a_x, b_x, a_y, b_y = get_calibration(img_bgr)

    for name in ("green", "blue"):
        lo, hi = COLOR_RANGES[name]
        mask = color_mask(img_bgr, lo, hi)
        rows = extract_segments(mask, a_x, b_x, a_y, b_y)
        if not rows:
            print(f"[{name}] 未匹配到像素，请检查 COLOR_RANGES")
            continue
        
        # 输出纯数据
        write_two_cols(OUTPUT_FILES[name], rows)
        
        # 输出并排对比图
        save_comparison(img_bgr, rows, COMPARE_FILES[name], name)
        
        xs = np.array([r[0] for r in rows])
        ys = np.array([r[1] for r in rows])
        print(f"[{name}] points={len(rows)}, "
              f"x∈[{xs.min():.6g},{xs.max():.6g}], "
              f"y∈[{ys.min():.6g},{ys.max():.6g}]")


if __name__ == "__main__":
    main()