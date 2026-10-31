"""Rebuild Glenzer A/B databases with measured SIF; preserve the Gaussian databases.

Examples:
  python run_glz_sif.py --limit 1
  python run_glz_sif.py --pilot
  python run_glz_sif.py

The SIF input axis defaults to energy-loss, as confirmed for sif_0.txt.
It is reflected into draft's redshift-negative photon-shift axis internally.
No automatic peak alignment or Gaussian width fitting is performed.
"""
import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import draft_sif as engine
from sif_response import DEFAULT_SIF, ROOT, MeasuredSIF

THETA = 40.0
E_LO, E_HI, NW = -70.0, 50.0, 1200
TE_GRID = list(range(1, 32))
NE_GRID = [1e29+j*5e28 for j in range(39)]
BASE = dict(E0_keV=2.96, ZA=4.0, Zf=2.2, ELECTRON_METHOD="integral",
            NU_MODE="const", nu_rad_s=5e14, SII_MODE="dh", USE_CHIHARA_ION=True)


def filename(te, j):
    return f"case_T{te:02d}_N{j:02d}.npz"


def atomic_npz(path, **data):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **data)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sif", type=Path, default=DEFAULT_SIF)
    parser.add_argument("--sif-axis", choices=["photon-shift", "energy-loss"], default="energy-loss")
    parser.add_argument("--negative", choices=["clip", "error"], default="clip")
    parser.add_argument("--output-root", type=Path, default=ROOT/"database")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--step", type=float, help="Maximum source step [eV]; smaller values test convergence")
    parser.add_argument("--te", type=int, nargs="+", help="Selected temperatures from 1..31 eV")
    parser.add_argument("--j", type=int, nargs="+", help="Selected density indices from 0..38")
    parser.add_argument("--only-b", action="store_true", help="Write/check B only; leave A untouched")
    parser.add_argument("--rebuild-b", action="store_true",
                        help="With --only-b, rebuild existing B files (full grid only)")
    args = parser.parse_args(argv)
    if args.rebuild_b and (not args.only_b or args.pilot or args.limit is not None or args.te or args.j):
        parser.error("--rebuild-b requires --only-b and the full grid (no pilot/limit/te/j)")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.step is not None and (not np.isfinite(args.step) or args.step <= 0):
        parser.error("--step must be finite and positive")
    if args.pilot and (args.te or args.j):
        parser.error("--pilot cannot be combined with --te/--j")
    if args.te and any(te not in TE_GRID for te in args.te):
        parser.error("--te must be within 1..31")
    if args.j and any(j not in range(39) for j in args.j):
        parser.error("--j must be within 0..38")
    sif = MeasuredSIF.load(args.sif, axis=args.sif_axis, negative=args.negative)
    da, db = args.output_root/"db_glz_a_sif", args.output_root/"db_glz_b_sif"
    source_files = ["physical_core.py", "draft_sif.py", "sif_response.py", "run_glz_sif.py"]
    run = dict(schema_version=1, base_physics=BASE, theta_deg=THETA,
               E_lo=E_LO, E_hi=E_HI, nw=NW, source_step_requested_eV=args.step,
               sif=sif.metadata,
               code_sha256={name: sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files},
               normalization_a="total peak = 1; components share total scale",
               normalization_b="raw free peak = 1; broadened free normalized AFTER convolution to its own peak = 1",
               gaussian_fwhm_used=False)
    signature = sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
    # A different kernel/axis/code/grid cannot silently resume into this database.
    directories = (db,) if args.only_b else (da, db)
    for directory in directories:
        manifest = directory/"run_config.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if previous.get("run_signature") != signature and not (args.rebuild_b and directory == db):
                parser.error(f"Run configuration differs in {directory}; select a fresh --output-root")
        elif directory.exists() and any(directory.glob("case_*.npz")) and not args.rebuild_b:
            parser.error(f"Unidentified existing database in {directory}; select a fresh --output-root")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        (directory/"run_config.json").write_text(
            json.dumps(dict(run_signature=signature, **run), ensure_ascii=False, indent=2), encoding="utf-8")
        np.savetxt(directory/"sif_kernel_used.txt", np.column_stack((sif.energy, sif.response)),
                   header="photon_shift_eV response_per_eV; area normalized; zero outside support")
    if args.pilot:
        cases = [(12, j) for j in range(39)]+[(1, 0), (1, 38), (31, 0), (31, 38)]
    else:
        cases = [(te, j) for te in sorted(set(args.te or TE_GRID))
                 for j in sorted(set(args.j or range(39)))]
    if args.limit is not None:
        cases = cases[:args.limit]
    print(json.dumps(sif.metadata, ensure_ascii=False), flush=True)
    print(f"Selected {len(cases)} cases; outputs: {', '.join(map(str, directories))}", flush=True)
    fields = ["Te_eV", "ne_m-3", "j", "status", "file_a", "file_b", "npts", "sec", "error"]
    report = args.output_root/"db_glz_sif_manifest.csv"
    failed = 0
    for i, (te, j) in enumerate(cases, 1):
        start = time.monotonic()
        fn = filename(te, j)
        row = dict(Te_eV=te, **{"ne_m-3": NE_GRID[j]}, j=j, file_a="" if args.only_b else str(da/fn),
                   file_b=str(db/fn), error="")
        try:
            existing = []
            for folder in directories:
                path = folder/fn
                if path.exists() and not (args.rebuild_b and folder == db):
                    with np.load(path, allow_pickle=False) as d:
                        if json.loads(str(d["meta_json"])).get("run_signature") != signature:
                            raise ValueError(f"Case configuration mismatch: {path}")
                        required = ["S_tot", "S_elastic", "S_inelastic"] if folder == da else ["S_free", "S_free_broadened"]
                        for key in required:
                            if d[key].shape != d["E_eV"].shape or not np.isfinite(d[key]).all():
                                raise ValueError(f"Invalid stored array: {path}:{key}")
                        existing.append(len(d["E_eV"]))
            if len(existing) == len(directories):
                if len(set(existing)) != 1:
                    raise ValueError("A/B grid sizes differ")
                row.update(status="skipped", npts=existing[0])
            else:
                cfg = engine.PhysicsConfig(Te_eV=float(te), ne=NE_GRID[j], **BASE)
                result = engine.compute_components(THETA, E_LO, E_HI, NW,
                                                   config=cfg, sif=sif, step=args.step)
                total_scale = float(result["S_tot_observed"].max())
                free_scale = float(result["S_free_raw"].max())
                free_observed_scale = float(result["S_free_observed"].max())
                if (not np.isfinite([total_scale, free_scale, free_observed_scale]).all()
                        or min(total_scale, free_scale, free_observed_scale) <= 0):
                    raise ValueError("Cannot normalize a nonpositive spectrum")
                physics_meta = asdict(cfg)
                physics_meta.pop("INSTRUMENT_FWHM_eV")  # unused inherited Gaussian parameter
                meta = dict(**physics_meta, theta_deg=THETA, E_lo=E_LO, E_hi=E_HI, nw=NW,
                            k=result["k"], j=j, run_signature=signature, sif=sif.metadata,
                            instrument_model="measured-SIF", gaussian_fwhm_used=False,
                            actual_nu_rad_s=result["nu_rad_s"], normalize=True,
                            free_raw_peak_before_normalization=free_scale,
                            free_broadened_peak_before_normalization=free_observed_scale,
                            total_peak_before_normalization=total_scale,
                            source_step_eV=result["source_step_eV"],
                            source_range_eV=result["source_range_eV"],
                            note_a=run["normalization_a"], note_b=run["normalization_b"])
                common = dict(E_eV=result["E_eV"], meta_json=json.dumps(meta),
                              rms_valid_mask=result["rms_valid_mask"])
                if not args.only_b:
                    atomic_npz(da/fn, **common, S_tot=result["S_tot_observed"]/total_scale,
                               S_elastic=result["S_elastic_observed"]/total_scale,
                               S_inelastic=cfg.Zf*result["S_free_observed"]/total_scale)
                atomic_npz(db/fn, **common, S_free=result["S_free_raw"]/free_scale,
                           S_free_broadened=result["S_free_observed"]/free_observed_scale)
                row.update(status="ok", npts=len(result["E_eV"]))
        except Exception as exc:
            failed += 1
            row.update(status="failed", npts="", error=str(exc))
        row["sec"] = round(time.monotonic()-start, 3)
        new = not report.exists()
        with report.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(f"{i}/{len(cases)} {fn}: {row['status']} ({row['sec']}s) {row['error']}", flush=True)
    print(f"Finished: {len(cases)-failed} successful/skipped; {failed} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
