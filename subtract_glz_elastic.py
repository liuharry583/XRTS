"""db_glz_a 的总谱 -> 拟合并扣除零频附近的高斯弹性峰。

运行：python subtract_glz_elastic.py
可选：--input PATH --output PATH --fwhm 8.7 --fit-window 13.05 --limit 3
模型 A*exp(-(E-center)^2/(2*sigma^2)) + b0 + b1*E。
FWHM 固定为生成库的仪器宽度，center 限制在 ±2 eV；只扣高斯，
不扣背景项，不裁负值，不重新归一化。剩余项是拟合残差，不是真值分离。
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "database/db_glz_a"
OUTPUT_DIR = ROOT / "database/db_glz_a_inelastic"
INSTRUMENT_FWHM_eV = 8.7
FIT_WINDOW_eV = 13.05  # ±1.5 FWHM；线性背景减少非弹性尾部对拟合的影响


def subtract_elastic(energy, total, fwhm=INSTRUMENT_FWHM_eV, fit_window=FIT_WINDOW_eV):
    energy, total = np.asarray(energy, float), np.asarray(total, float)
    if (energy.ndim != 1 or total.shape != energy.shape
            or not np.isfinite(energy).all() or not np.isfinite(total).all()
            or not np.isfinite([fwhm, fit_window]).all()
            or fwhm <= 0 or fit_window < fwhm):
        raise ValueError("Expected matching finite spectra, positive FWHM and window >= FWHM")
    order = np.argsort(energy)
    x_all, y_all = energy[order], total[order]
    if np.any(np.diff(x_all) <= 0):
        raise ValueError("Energy coordinates must be unique")
    mask = np.abs(x_all) <= fit_window
    x, y = x_all[mask], y_all[mask]
    if len(x) < 12 or x.min() > -fwhm or x.max() < fwhm:
        raise ValueError("Need data on both sides of zero covering at least +/- FWHM")
    sigma = fwhm / np.sqrt(8 * np.log(2))
    scale = float(np.max(np.abs(y)))
    if scale <= 0:
        raise ValueError("Cannot fit a zero spectrum")
    yn = y / scale
    edge = np.abs(x) >= 0.75 * fit_window
    slope, baseline = np.polyfit(x[edge] / fit_window, yn[edge], 1)
    residual = yn - baseline - slope * x / fit_window
    central = np.abs(x) <= min(2.0, fwhm / 2)
    center = float(x[central][np.argmax(residual[central])])
    amplitude = max(float(residual[central].max()), 1e-8)

    def model(p, e):
        a, mu, b, m = p
        return a * np.exp(-0.5 * ((e - mu) / sigma) ** 2) + b + m * e / fit_window

    result = least_squares(
        lambda p: model(p, x) - yn, [amplitude, center, baseline, slope],
        bounds=([0, -2, -np.inf, -np.inf], [np.inf, 2, np.inf, np.inf]),
        loss="soft_l1", f_scale=0.02, max_nfev=2000,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Gaussian fit failed: {result.message}")
    a, mu, b, m = result.x
    if a <= 1e-8:
        raise ValueError("No identifiable positive elastic peak")
    elastic = a * scale * np.exp(-0.5 * ((energy - mu) / sigma) ** 2)
    remaining = total - elastic
    relative_fit_rmse = float(np.sqrt(np.mean((model(result.x, x) - yn) ** 2)))
    warnings = []
    if abs(mu) >= 1.99:
        warnings.append("center_at_bound")
    if relative_fit_rmse > 0.03:
        warnings.append("large_fit_residual")
    if remaining.min() < -0.03 * scale:
        warnings.append("negative_residual_below_minus_3percent_peak")
    fit = dict(amplitude=float(a * scale), center_eV=float(mu), fwhm_eV=fwhm,
               baseline=float(b * scale), slope_per_eV=float(m * scale / fit_window),
               fit_window_eV=fit_window, relative_fit_rmse=relative_fit_rmse,
               minimum_remaining=float(remaining.min()), warnings=warnings,
               renormalized=False, negative_values_clipped=False)
    return remaining, elastic, fit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fwhm", type=float, default=INSTRUMENT_FWHM_eV)
    parser.add_argument("--fit-window", type=float, default=FIT_WINDOW_eV)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from source database")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    files = sorted(args.input.glob("case_*.npz"))
    if not files:
        parser.error("No case_*.npz files found in input")
    if args.limit:
        files = files[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(files, 1):
        row = dict(file=path.name, status="ok", error="")
        try:
            with np.load(path, allow_pickle=False) as z:
                data = {key: z[key] for key in z.files}
            if "S_inelastic" in data:
                raise ValueError("Input already contains subtraction output")
            meta = json.loads(str(data.get("meta_json", "{}")))
            stored_fwhm = meta.get("INSTRUMENT_FWHM_eV")
            if stored_fwhm is not None and not np.isclose(stored_fwhm, args.fwhm):
                raise ValueError(f"Stored FWHM {stored_fwhm} differs from fit FWHM {args.fwhm}")
            remaining, elastic, fit = subtract_elastic(
                data["E_eV"], data["S_tot"], args.fwhm, args.fit_window)
            fit["source"] = str(path.resolve())
            np.savez_compressed(args.output / path.name, **data, S_inelastic=remaining,
                                S_elastic_fit=elastic, fit_json=json.dumps(fit))
            row.update({key: fit[key] for key in ("amplitude", "center_eV",
                       "relative_fit_rmse", "minimum_remaining")})
            row["warnings"] = ";".join(fit["warnings"])
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        rows.append(row)
        if index % 200 == 0 or index == len(files):
            print(f"{index}/{len(files)} processed", flush=True)
    fields = ["file", "status", "error", "amplitude", "center_eV",
              "relative_fit_rmse", "minimum_remaining", "warnings"]
    with (args.output / "fit_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failed = sum(row["status"] == "failed" for row in rows)
    flagged = sum(bool(row.get("warnings")) for row in rows)
    print(f"Output: {args.output}; success={len(rows)-failed}, failed={failed}, flagged={flagged}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
