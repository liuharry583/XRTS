"""Compare three already normalized experimental spectra with measured-SIF databases.

Run: python rms_sif_contours.py
Only NumPy and Matplotlib are required. No physics recalculation or convolution.
See rms_sif_usage.md for definitions, assumptions and command-line examples.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

# User configuration; relative paths are resolved against this script's directory.
ROOT = Path(__file__).resolve().parent
DATABASE_A = "database/db_glz_a_sif"
DATABASE_B = "database/db_glz_b_sif"
EXPERIMENT_TOTAL = "exp_elastic_1_negative.txt"
EXPERIMENT_INELASTIC_TOTAL = "exp_noelastic_1_negative.txt"
EXPERIMENT_INELASTIC_PEAK = "exp_noelastic_2_negative.txt"
OUTPUT_DIRECTORY = "rms_sif_results"
ENERGY_MIN_EV = None  # Photon shift, energy loss is NEGATIVE; None = full overlap.
ENERGY_MAX_EV = None
DUPLICATES = "mean"  # mean: one mean intensity per exact energy; keep: all rows.

GROUPS = {
    "total": ("S_tot", "Total spectrum / total-peak scale"),
    "inelastic_total": ("S_inelastic", "Inelastic spectrum / total-peak scale"),
    "inelastic_peak": ("S_free_broadened", "Inelastic spectrum / inelastic-peak scale"),
}
PHYSICS_KEYS = (
    "E0_keV", "theta_deg", "ZA", "Zf", "NU_MODE", "SII_MODE",
    "nu_rad_s", "NU_FRAC", "actual_nu_rad_s", "ELECTRON_METHOD",
    "USE_CHIHARA_ION", "instrument_model", "gaussian_fwhm_used",
)


def resolve(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def signature(meta):
    # A and rebuilt B have different code/run signatures, but must share physics/SIF.
    values = {k: meta[k] for k in PHYSICS_KEYS}
    values["sif_sha256"] = meta["sif"]["sha256"]
    values["sif_input_axis"] = meta["sif"]["input_axis"]
    values["sif_convolution_axis"] = meta["sif"]["convolution_axis"]
    return values


@dataclass
class Case:
    path: Path
    te: float
    ne: float
    energy: np.ndarray
    valid: np.ndarray
    spectra: dict
    meta: dict


def load_database(directory, fields):
    """Read actual NPZ files, never the append-only shared manifest."""
    paths = sorted(directory.glob("case_*.npz"))
    if not paths:
        raise ValueError(f"No case_*.npz files in {directory}")
    cases, seen, reference = [], set(), None
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta_json"]))
                energy = np.asarray(data["E_eV"], dtype=float)
                raw_mask = data["rms_valid_mask"]
                spectra = {field: np.asarray(data[field], dtype=float) for field in fields}
                if energy.ndim != 1 or len(energy) < 2 or not np.isfinite(energy).all():
                    raise ValueError("Invalid energy axis")
                if raw_mask.shape != energy.shape or not np.isin(raw_mask, [0, 1]).all():
                    raise ValueError("Invalid rms_valid_mask")
                if any(y.shape != energy.shape or not np.isfinite(y).all() for y in spectra.values()):
                    raise ValueError("Invalid spectrum shape or nonfinite intensity")
                if "S_tot" in fields:
                    if not np.allclose(data["S_tot"], data["S_elastic"] + data["S_inelastic"],
                                       rtol=1e-10, atol=1e-12):
                        raise ValueError("S_tot != S_elastic + S_inelastic")
                normalized = spectra["S_tot" if "S_tot" in fields else "S_free_broadened"]
                if not np.isclose(normalized.max(), 1.0, rtol=1e-8, atol=1e-10):
                    raise ValueError("Stored peak is not 1; use the correctly normalized database")
            current = signature(meta)
            if reference is None:
                reference = current
            if current != reference:
                raise ValueError("Mixed physics or SIF within this database")
            if current["instrument_model"] != "measured-SIF" or current["sif_convolution_axis"] != "photon-shift":
                raise ValueError("Expected measured-SIF spectra on the photon-shift axis")
            if meta.get("normalize") is not True:
                raise ValueError("Database spectra must already be normalized")
            te, ne = float(meta["Te_eV"]), float(meta.get("ne", meta.get("ne_m3")))
            if not np.isfinite([te, ne]).all() or te <= 0 or ne <= 0:
                raise ValueError("Invalid temperature or electron density")
            if (te, ne) in seen:
                raise ValueError(f"Duplicate parameter point: {(te, ne)}")
            seen.add((te, ne))
            order = np.argsort(energy, kind="stable")
            energy = energy[order]
            if np.any(np.diff(energy) <= 0):
                raise ValueError("Duplicate theory energy coordinates")
            cases.append(Case(path, te, ne, energy, raw_mask[order].astype(bool),
                              {k: y[order] for k, y in spectra.items()}, meta))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}: {error}") from error
    temperatures = sorted({c.te for c in cases})
    densities = sorted({c.ne for c in cases})
    if len(temperatures) < 2 or len(densities) < 2:
        raise ValueError(f"{directory}: need at least 2 temperatures and 2 densities for contours")
    if len(cases) != len(temperatures) * len(densities):
        raise ValueError(f"{directory}: incomplete rectangular grid; refusing to fill missing cases")
    return cases, reference


def read_experiment(path, duplicates):
    data = np.loadtxt(path, encoding="utf-8-sig", ndmin=2)
    if data.shape[1] != 2 or len(data) < 3 or not np.isfinite(data).all():
        raise ValueError(f"{path}: expected at least 3 finite rows, exactly 2 columns (eV, intensity)")
    data = data[np.argsort(data[:, 0], kind="stable")]
    raw_x, raw_y = data.T
    x, inverse, counts = np.unique(raw_x, return_inverse=True, return_counts=True)
    if len(x) < 3:
        raise ValueError(f"{path}: need at least 3 distinct energies")
    report = dict(path=str(path), sha256=sha256(path), raw_rows=len(data),
                  unique_energies=len(x), duplicate_rows=len(data) - len(x),
                  duplicate_policy=duplicates, raw_energy_range_eV=[float(raw_x[0]), float(raw_x[-1])],
                  raw_intensity_range=[float(raw_y.min()), float(raw_y.max())])
    if duplicates == "mean":
        y = np.bincount(inverse, weights=raw_y) / counts
    else:
        x, y, counts = raw_x.copy(), raw_y.copy(), np.ones(len(raw_x), dtype=int)
    return x, y, counts, report


def covered_points(case, x):
    """Both interpolation endpoints must be valid; never bridge invalid mask holes."""
    e = case.energy
    right = np.searchsorted(e, x, side="left").clip(0, len(e) - 1)
    left = (right - 1).clip(0)
    left = np.where(e[right] == x, right, left)
    return (x >= e[0]) & (x <= e[-1]) & case.valid[left] & case.valid[right]


def rms_values(predictions, experiment):
    return np.sqrt(np.mean((predictions - experiment) ** 2, axis=-1))


def plot_contour(grid, temperatures, densities, report, path, levels, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    # Paper-like dark low-RMS island, teal middle, pale large residuals.
    cmap = LinearSegmentedColormap.from_list("rms_teal", ["#183b53", "#167c80", "#00c9ae", "#a9eef0", "#f1fbfc"])
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "axes.spines.top": True,
                         "axes.spines.right": True, "figure.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(7.6, 6.8), layout="constrained")
        low, high = float(grid.min()), float(grid.max())
        if high <= low:
            high = low + max(abs(low) * 0.01, 1e-8)
        boundaries = np.linspace(low, high, levels + 1)
        filled = ax.contourf(temperatures, densities / 1e29, grid, levels=boundaries, cmap=cmap)
        lines = ax.contour(temperatures, densities / 1e29, grid,
                           levels=boundaries[1:-1:2], colors="#22595e", linewidths=0.55, alpha=0.8)
        ax.clabel(lines, inline=True, fontsize=8, fmt="%.3f")
        best = report["best"]
        ax.axvline(best["Te_eV"], color="white", ls="-.", lw=1.0, alpha=0.9)
        ax.axhline(best["ne_m3"] / 1e29, color="white", ls="-.", lw=1.0, alpha=0.9)
        ax.scatter([best["Te_eV"]], [best["ne_m3"] / 1e29], marker="*", s=160,
                   facecolor="#ffda76", edgecolor="#172c3a", linewidth=0.9, zorder=5)
        ax.set(xlabel=r"Electron temperature $T_e$ (eV)",
               ylabel=r"Electron density $n_e$ ($10^{29}$ m$^{-3}$)",
               title=GROUPS[report["group"]][1] + "\nRMS = sqrt(mean((theory - experiment)^2))")
        ax.tick_params(direction="in", top=True, right=True)
        fig.colorbar(filled, ax=ax, label="RMS (stored normalized-intensity units)", format="%.3f", shrink=0.88)
        annotation = (f"Grid minimum: Te={best['Te_eV']:g} eV, ne={best['ne_m3']/1e29:g} x 10^29 m^-3\n"
                      f"RMS={best['rms']:.6f}; N={report['used_points']}; "
                      f"shift=[{report['used_energy_range_eV'][0]:.2f}, {report['used_energy_range_eV'][1]:.2f}] eV")
        if best["on_grid_boundary"]:
            annotation += "\nMinimum is on the scanned parameter boundary."
        fig.supxlabel(annotation, fontsize=9)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)


def scan_group(group, cases, experiment, all_cases, args, output):
    field = GROUPS[group][0]
    x, y, counts, report = experiment
    keep = np.ones(len(x), dtype=bool)
    if args.energy_min is not None:
        keep &= x >= args.energy_min
    if args.energy_max is not None:
        keep &= x <= args.energy_max
    window_count = int(keep.sum())
    # The same fixed experiment points are used for EVERY parameter point.
    # Include both A and B support so the three groups share valid theory coverage.
    for case in all_cases:
        keep &= covered_points(case, x)
    if keep.sum() < 3 or np.ptp(x[keep]) <= 0:
        raise ValueError(f"{group}: fewer than 3 points in the requested common valid support")
    x, y, counts = x[keep], y[keep], counts[keep]
    predictions = np.array([np.interp(x, c.energy, c.spectra[field]) for c in cases])
    values = rms_values(predictions, y)
    temperatures = np.array(sorted({c.te for c in cases}))
    densities = np.array(sorted({c.ne for c in cases}))
    ti = {t: i for i, t in enumerate(temperatures)}
    ni = {n: i for i, n in enumerate(densities)}
    grid = np.full((len(densities), len(temperatures)), np.nan)
    for case, value in zip(cases, values):
        grid[ni[case.ne], ti[case.te]] = value
    index = int(np.argmin(values))
    case = cases[index]
    best = dict(Te_eV=case.te, ne_m3=case.ne, ne_cm3=case.ne / 1e6,
                rms=float(values[index]), file=str(case.path),
                on_grid_boundary=bool(case.te in (temperatures[0], temperatures[-1])
                                     or case.ne in (densities[0], densities[-1])),
                tied_grid_minima=int(np.count_nonzero(values == values[index])))
    report.update(group=group, database=str(case.path.parent), field=field,
                  requested_energy_range_eV=[args.energy_min, args.energy_max],
                  points_in_requested_window=window_count,
                  removed_outside_common_theory_support=window_count - len(x),
                  used_points=len(x), used_energy_range_eV=[float(x[0]), float(x[-1])],
                  used_intensity_range=[float(y.min()), float(y.max())],
                  grid_points=len(cases), best=best, best_case_metadata=case.meta)
    prefix = output / group
    np.savez_compressed(str(prefix) + "_grid.npz", Te_eV=temperatures, ne_m3=densities,
                        rms=grid, experiment_energy_eV=x, experiment_intensity=y,
                        best_theory=predictions[index], experiment_source_rows=counts)
    write_csv(Path(str(prefix) + "_scan.csv"), ["Te_eV", "ne_m3", "RMS", "N_points", "case_file"],
              ([c.te, c.ne, float(v), len(x), c.path.name] for c, v in zip(cases, values)))
    write_csv(Path(str(prefix) + "_best_fit.csv"),
              ["photon_shift_eV", "experiment", "theory", "theory_minus_experiment", "source_rows"],
              zip(x, y, predictions[index], predictions[index] - y, counts))
    write_csv(Path(str(prefix) + "_best_theory_full.csv"),
              ["photon_shift_eV", field, "rms_valid_mask"],
              zip(case.energy, case.spectra[field], case.valid.astype(int)))
    plot_contour(grid, temperatures, densities, report,
                 Path(str(prefix) + "_rms_contour.png"), args.levels, args.dpi)
    print(f"{group}: Te={case.te:g} eV, ne={case.ne:.6g} m^-3, "
          f"RMS={values[index]:.8f}, N={len(x)}, boundary={best['on_grid_boundary']}", flush=True)
    return report


def plot_best_fits(reports, output, dpi):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), layout="constrained", sharex=True)
    for ax, report in zip(axes, reports):
        group = report["group"]
        with np.load(output / f"{group}_grid.npz", allow_pickle=False) as data:
            x, y, theory = data["experiment_energy_eV"], data["experiment_intensity"], data["best_theory"]
        ax.plot(x, y, color="#262b33", lw=0.85, label="Experiment (RMS input)")
        ax.plot(x, theory, color="#d04a39", lw=1.5, label="Best theory on database grid")
        ax.axhline(0, color="0.7", lw=0.5)
        best = report["best"]
        ax.set(ylabel="Stored intensity", title=GROUPS[group][1] +
               f"; Te={best['Te_eV']:g} eV, ne={best['ne_m3']/1e29:g} x 10^29 m^-3, RMS={best['rms']:.5f}")
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Photon energy shift (eV); energy loss is negative")
    fig.savefig(output / "best_fit_comparison.png", dpi=dpi)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-a", default=DATABASE_A)
    parser.add_argument("--db-b", default=DATABASE_B)
    parser.add_argument("--exp-total", default=EXPERIMENT_TOTAL)
    parser.add_argument("--exp-inelastic-total", default=EXPERIMENT_INELASTIC_TOTAL)
    parser.add_argument("--exp-inelastic-peak", default=EXPERIMENT_INELASTIC_PEAK)
    parser.add_argument("--output-dir", default=OUTPUT_DIRECTORY)
    parser.add_argument("--energy-min", type=float, default=ENERGY_MIN_EV, help="Photon-shift lower bound in eV")
    parser.add_argument("--energy-max", type=float, default=ENERGY_MAX_EV, help="Photon-shift upper bound in eV")
    parser.add_argument("--duplicates", choices=("mean", "keep"), default=DUPLICATES)
    parser.add_argument("--levels", type=int, default=18, help="Number of linear contour intervals")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        if args.levels < 3 or args.dpi < 50:
            raise ValueError("Need --levels >= 3 and --dpi >= 50")
        for bound in (args.energy_min, args.energy_max):
            if bound is not None and not np.isfinite(bound):
                raise ValueError("Energy bounds must be finite")
        if args.energy_min is not None and args.energy_max is not None and args.energy_min >= args.energy_max:
            raise ValueError("--energy-min must be smaller than --energy-max")
        import matplotlib
        matplotlib.use("Agg")  # Batch execution without GUI windows.
        print("Reading and validating measured-SIF database files...", flush=True)
        a, sig_a = load_database(resolve(args.db_a), ("S_tot", "S_inelastic"))
        b, sig_b = load_database(resolve(args.db_b), ("S_free_broadened",))
        if sig_a != sig_b:
            raise ValueError("A and B physics / instrument metadata differ")
        if {(c.te, c.ne) for c in a} != {(c.te, c.ne) for c in b}:
            raise ValueError("A and B parameter grids differ")
        experiments = [read_experiment(resolve(p), args.duplicates) for p in
                       (args.exp_total, args.exp_inelastic_total, args.exp_inelastic_peak)]
        output = resolve(args.output_dir)
        # Prevent output inside either input database.
        for database in (resolve(args.db_a), resolve(args.db_b)):
            if output == database or database in output.parents:
                raise ValueError("Output directory must be outside the input databases")
        output.mkdir(parents=True, exist_ok=True)
        reports = [scan_group(g, cases, exp, a + b, args, output)
                   for g, cases, exp in zip(GROUPS, (a, a, b), experiments)]
        plot_best_fits(reports, output, args.dpi)
        database_provenance = {}
        for label, cases in (("a", a), ("b", b)):
            digest = hashlib.sha256()
            for case in cases:
                digest.update(case.path.name.encode("utf-8") + b"\0" + bytes.fromhex(sha256(case.path)))
            config = cases[0].path.parent / "run_config.json"
            database_provenance[label] = dict(path=str(cases[0].path.parent), files=len(cases),
                sorted_filenames_and_file_hashes_sha256=digest.hexdigest(),
                run_config_sha256=sha256(config) if config.exists() else None,
                run_signatures=sorted({c.meta.get("run_signature", "") for c in cases}),
                source_step_eV_range=[min(float(c.meta["source_step_eV"]) for c in cases),
                                      max(float(c.meta["source_step_eV"]) for c in cases)])
        summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                       script_sha256=sha256(Path(__file__).resolve()),
                       numpy_version=np.__version__, matplotlib_version=matplotlib.__version__,
                       rms_definition="sqrt(mean((theory_interpolated_to_experiment - experiment)**2))",
                       weighting="equal weight per retained row after duplicate handling",
                       energy_axis="photon shift (eV); energy loss is negative",
                       interpolation="linear; no extrapolation; both bracketing points must have rms_valid_mask=True",
                       normalization="stored intensities unchanged on both sides",
                       fitted_nuisance_parameters=[], physics=sig_a,
                       database_provenance=database_provenance,
                       plot_note="Contours interpolate between scanned grid points; color scales are independent. No confidence intervals.",
                       groups=reports)
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(output / "best_fits.csv", ["group", "field", "Te_eV", "ne_m3", "ne_cm3", "RMS", "N_points", "on_grid_boundary"],
                  ([r["group"], r["field"], r["best"]["Te_eV"], r["best"]["ne_m3"],
                    r["best"]["ne_cm3"], r["best"]["rms"], r["used_points"], r["best"]["on_grid_boundary"]]
                   for r in reports))
        print(f"Results saved to {output}", flush=True)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
