"""将实验光谱从“能量损失为正”转换为“能量损失为负”。

直接运行：python flip_exp_elastic_axis.py
默认读取本目录 exp_elastic_1.txt，另存 exp_elastic_1_negative.txt。
操作只有 x_new=-x_old，再把(x,y)配对按新x升序稳定排序。
不改y、不裁负数、不归一化、不合并重复坐标，不覆盖原文件。
可指定：--input PATH --output PATH；输出已存在时拒绝覆盖。
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"exp_elastic_1.txt")
    parser.add_argument("--output", type=Path, default=ROOT/"exp_elastic_1_negative.txt")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("输出路径必须与输入不同，避免覆盖原数据")
    data = np.loadtxt(args.input, encoding="utf-8-sig", ndmin=2)
    if data.shape[1] != 2 or not len(data) or not np.isfinite(data).all():
        parser.error("输入必须是有限数值的两列：能量/eV、强度")
    result = data.copy()
    result[:, 0] *= -1
    result = result[np.argsort(result[:, 0], kind="stable")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 保留纯两列格式；17位有效数字保证浮点读回不损失精度。
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        np.savetxt(stream, result, fmt="%.17g", delimiter="\t")
    print(f"Saved {len(result)} rows: {args.output.resolve()}")
    print(f"New x range [eV]: {result[0, 0]:g} .. {result[-1, 0]:g}")


if __name__ == "__main__":
    main()
