"""Apply sif_0.txt to RAW db_glz_b S_free, then normalize the broadened peak to 1."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sif_response import DEFAULT_SIF, ROOT, MeasuredSIF


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"database/db_glz_b")
    parser.add_argument("--output", type=Path, default=ROOT/"database/db_glz_b_sif_postprocessed")
    parser.add_argument("--sif", type=Path, default=DEFAULT_SIF)
    parser.add_argument("--sif-axis", choices=["photon-shift", "energy-loss"], default="energy-loss")
    parser.add_argument("--negative", choices=["clip", "error"], default="clip")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from the source database")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    files = sorted(args.input.glob("case_*.npz"))
    if not files:
        parser.error("No case_*.npz files found")
    sif = MeasuredSIF.load(args.sif, axis=args.sif_axis, negative=args.negative)
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(sif.metadata, ensure_ascii=False), flush=True)
    rows = []
    for path in files[:args.limit]:
        row = dict(file=path.name, status="ok", error="")
        try:
            target = args.output/path.name
            if target.exists():
                raise FileExistsError(f"Output exists; use a fresh directory: {target}")
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            if "S_free_broadened" in data or "processing_json" in data:
                raise ValueError("Input already processed; select the original raw B database")
            out, mask = sif.broaden_existing(data["E_eV"], data["S_free"])
            peak = float(out.max())
            if not np.isfinite(peak) or peak <= 0:
                raise ValueError("Cannot normalize a nonpositive/nonfinite broadened peak")
            out = out / peak
            processing = dict(operation="measured-SIF convolution", sif=sif.metadata,
                              source=str(path.resolve()), renormalized=True,
                              normalization="broaden first, then divide by broadened peak",
                              broadened_peak_before_normalization=peak,
                              source_outside_window="zero", input_field="S_free",
                              output_field="S_free_broadened")
            np.savez_compressed(target, **data, S_free_broadened=out,
                                rms_valid_mask=mask, processing_json=json.dumps(processing))
            row.update(peak_before=float(data["S_free"].max()), peak_after=float(out.max()),
                       valid_points=int(mask.sum()), npts=len(out))
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"{len(rows)} cases processed", flush=True)
    with (args.output/"processing_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "status", "error", "peak_before",
                                              "peak_after", "valid_points", "npts"])
        writer.writeheader()
        writer.writerows(rows)
    failed = sum(r["status"] != "ok" for r in rows)
    print(f"Output: {args.output}; success={len(rows)-failed}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
