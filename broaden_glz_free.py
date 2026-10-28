"""db_glz_b 的未展宽 S_free -> 8.7 eV 高斯卷积；不重新归一化。

运行：python broaden_glz_free.py
可选：--input PATH --output PATH --fwhm 8.7 --limit 3
输出保留原字段，新增 S_free_broadened；输入文件不改动。
有限能区外按零处理，距边界 4 sigma 内标为不建议用于 RMS。
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "database/db_glz_b"
OUTPUT_DIR = ROOT / "database/db_glz_b_broadened"
INSTRUMENT_FWHM_eV = 8.7


def broaden_spectrum(energy, spectrum, fwhm=INSTRUMENT_FWHM_eV):
    """式(54)的能量坐标形式，梯形积分权重；返回原顺序与边缘掩码。"""
    energy, spectrum = np.asarray(energy, float), np.asarray(spectrum, float)
    if (energy.ndim != 1 or energy.size < 3 or spectrum.shape != energy.shape
            or not np.isfinite(energy).all() or not np.isfinite(spectrum).all()
            or not np.isfinite(fwhm) or fwhm <= 0):
        raise ValueError("Expected finite matching arrays and positive FWHM")
    order = np.argsort(energy)
    x, y = energy[order], spectrum[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Energy coordinates must be unique")
    sigma = fwhm / np.sqrt(8 * np.log(2))
    steps = np.diff(x)
    # 库中近零频删点造成的少量缺口用线性插值填回；不重算物理模型。
    count = int(np.ceil((x[-1] - x[0]) / min(np.median(steps), sigma / 8)))
    grid = np.linspace(x[0], x[-1], count + 1)
    dx = grid[1] - grid[0]
    sampled = np.interp(grid, x, y)
    sampled[[0, -1]] *= 0.5  # 源区间两端梯形权重
    radius = int(np.ceil(8 * sigma / dx))
    offsets = np.arange(-radius, radius + 1) * dx
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    blurred = fftconvolve(sampled, kernel, mode="same") * dx
    output = np.empty_like(y)
    output[order] = np.interp(x, grid, blurred)
    valid = (energy >= x[0] + 4 * sigma) & (energy <= x[-1] - 4 * sigma)
    return output, valid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fwhm", type=float, default=INSTRUMENT_FWHM_eV)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from source database")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not np.isfinite(args.fwhm) or args.fwhm <= 0:
        parser.error("--fwhm must be finite and positive")
    files = sorted(args.input.glob("case_*.npz"))
    if not files:
        parser.error("No case_*.npz files found in input")
    if args.limit:
        files = files[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(files, 1):
        row = {"file": path.name, "status": "ok", "error": ""}
        try:
            with np.load(path, allow_pickle=False) as z:
                data = {key: z[key] for key in z.files}
            if "S_free_broadened" in data:
                raise ValueError("Input already contains broadened output")
            out, mask = broaden_spectrum(data["E_eV"], data["S_free"], args.fwhm)
            processing = dict(operation="Gaussian convolution, Eq.54",
                              source=str(path.resolve()), input_field="S_free",
                              output_field="S_free_broadened",
                              fwhm_eV=args.fwhm, renormalized=False,
                              source_outside_window="zero", rms_edge_exclusion_sigma=4)
            np.savez_compressed(args.output / path.name, **data,
                                S_free_broadened=out, rms_valid_mask=mask,
                                processing_json=json.dumps(processing))
            row.update(peak_before=float(data["S_free"].max()),
                       peak_after=float(out.max()), npts=len(out))
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        rows.append(row)
        if index % 200 == 0 or index == len(files):
            print(f"{index}/{len(files)} processed", flush=True)
    keys = ["file", "status", "error", "peak_before", "peak_after", "npts"]
    with (args.output / "processing_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    failed = sum(row["status"] == "failed" for row in rows)
    print(f"Output: {args.output}; success={len(rows)-failed}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
