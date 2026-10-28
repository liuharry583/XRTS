# -*- coding: utf-8 -*-
"""
RPA 数据库批量跑批（Glenzer 口径双库，2026-10-26 批准口径）

引擎：~/workspace/user/files/draft.py（整谱同一高斯仪器展宽，已含组会任务1 的修正）
产物（同一次 compute 双库同源产出，均存 normalize=True 的归一谱）：
  - 库A（含 Sii）：db_glz_a/case_T{Te:02d}_N{j:02d}.npz 内 S_tot（全谱最大值归一为 1）
  - 库B（不含 Sii）：db_glz_b/同名 .npz 内 S_free（自身最大值归一为 1；注：引擎返回的
    S_free 为未展宽谱，展宽/卷积在 RMS 比较端处理）
清单：db_glz_manifest.csv（与本脚本同目录）

固定口径（Glenzer et al. 2007, PRL 98, 065002）：
  E0=2.96 keV, theta=40 deg, Zf=2.2, ZA=4 (Be),
  ELECTRON_METHOD="integral", NU_MODE="const" + nu_rad_s=5e14 s^-1（原文 nu_ei）,
  SII_MODE="dh", INSTRUMENT_FWHM_eV=8.7（抠图 SIF 实测 FWHM）
网格：Te = 1..31 eV（步长 1）；ne = 1e29 + j*5e28 m^-3, j=0..38（共 1209 case）
能量窗：E_lo=-70, E_hi=+50 eV（引擎约定，下移为负），nw=1200（步长 0.1 eV）

用法：
  python3 RPA数据库_批量跑批_glz.py --pilot   # 先导批：Te=12 整行 + 四角
  python3 RPA数据库_批量跑批_glz.py           # 全量（自动跳过已完成 case，可断点续跑）
"""
import argparse, csv, json, os, sys, time
import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

sys.path.insert(0, os.path.expanduser("~/workspace/user/files"))
import draft as pc

HERE = os.path.dirname(os.path.abspath(__file__))
DIR_A = os.path.join(HERE, "db_glz_a")
DIR_B = os.path.join(HERE, "db_glz_b")
MANIFEST = os.path.join(HERE, "db_glz_manifest.csv")

THETA = 40.0
E_LO, E_HI, NW = -70.0, 50.0, 1200
TE_GRID = list(range(1, 32))                       # 1..31 eV
NE_GRID = [1.0e29 + j * 5.0e28 for j in range(39)]  # 1e29 .. 2e30 m^-3

BASE = dict(E0_keV=2.96, ZA=4.0, Zf=2.2,
            ELECTRON_METHOD="integral", NU_MODE="const", nu_rad_s=5.0e14,
            SII_MODE="dh", INSTRUMENT_FWHM_eV=8.7)

FIELDS = ["Te_eV", "ne_m-3", "j", "status", "file_a", "file_b", "npts",
          "peakE_a_eV", "peakE_b_eV", "max_a", "max_b", "sec"]


def fname(te, j):
    return f"case_T{te:02d}_N{j:02d}.npz"


def done_set():
    done = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add((int(row["Te_eV"]), int(row["j"])))
    return done


def append_manifest(row):
    new = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_case(te, j, ne):
    cfg = pc.PhysicsConfig(Te_eV=float(te), ne=float(ne), **BASE)
    t0 = time.time()
    E, w, Sf, St, k = pc.compute(THETA, E_LO, E_HI, nw=NW, config=cfg, normalize=True)
    sec = time.time() - t0
    meta = dict(theta_deg=THETA, E0_keV=2.96, Te_eV=float(te), ne=float(ne),
                Zf=2.2, ZA=4.0, ELECTRON_METHOD="integral", NU_MODE="const",
                nu_rad_s=5.0e14, SII_MODE="dh", INSTRUMENT_FWHM_eV=8.7,
                E_lo=E_LO, E_hi=E_HI, nw=NW, k=float(k), normalize=True,
                note_a="S_tot normalized to its own peak (=1)",
                note_b="S_free normalized to its own peak (=1); unbroadened free spectrum as returned by draft.py compute")
    meta_json = json.dumps(meta)
    fn = fname(te, j)
    np.savez_compressed(os.path.join(DIR_A, fn), E_eV=E, S_tot=St, meta_json=meta_json)
    np.savez_compressed(os.path.join(DIR_B, fn), E_eV=E, S_free=Sf, meta_json=meta_json)
    return dict(Te_eV=te, **{"ne_m-3": f"{ne:.3e}"}, j=j, status="ok",
                file_a=fn, file_b=fn, npts=len(E),
                peakE_a_eV=f"{E[int(np.argmax(St))]:.2f}",
                peakE_b_eV=f"{E[int(np.argmax(Sf))]:.2f}",
                max_a=f"{St.max():.6f}", max_b=f"{Sf.max():.6f}",
                sec=f"{sec:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    os.makedirs(DIR_A, exist_ok=True)
    os.makedirs(DIR_B, exist_ok=True)
    if args.pilot:
        cases = [(12, j, NE_GRID[j]) for j in range(39)] + \
                [(1, 0, NE_GRID[0]), (1, 38, NE_GRID[38]),
                 (31, 0, NE_GRID[0]), (31, 38, NE_GRID[38])]
    else:
        cases = [(te, j, NE_GRID[j]) for te in TE_GRID for j in range(39)]
    done = done_set()
    todo = [c for c in cases if (c[0], c[1]) not in done]
    print(f"total {len(cases)}, already done {len(cases)-len(todo)}, to run {len(todo)}", flush=True)
    t_all = time.time()
    for n, (te, j, ne) in enumerate(todo, 1):
        try:
            row = run_case(te, j, ne)
        except Exception as ex:
            row = dict(Te_eV=te, **{"ne_m-3": f"{ne:.3e}"}, j=j, status=f"FAIL:{ex}",
                       file_a="", file_b="", npts="", peakE_a_eV="", peakE_b_eV="",
                       max_a="", max_b="", sec="")
            print("FAIL", te, j, ex, flush=True)
        append_manifest(row)
        if n % 20 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} done, elapsed {time.time()-t_all:.0f}s", flush=True)
    print(f"ALL DONE in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
