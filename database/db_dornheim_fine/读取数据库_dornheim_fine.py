# -*- coding: utf-8 -*-
"""db_dornheim_fine 读取器（Dornheim 专用细网格库）。

用法：
    from 读取数据库_dornheim_fine import load_case, grid_axes
    d = load_case(11.0, 1.55e30)     # Te[eV], ne[m^-3]，必须命中网格点
    d["E_eV"], d["S_free"], d["S_tot"], d["k"], d["meta"]

约定：E_eV 降序（+100 → -100 eV）；S_free/S_tot 与引擎 compute() 返回同义。
注意：ne >= 1.85e30 的 124 个 case 因引擎 ω≈0 掩膜（|ω|<=1e-3*ωp 的点被剔除）
在 E≈0 处少 2 个点（1998 点、中心间隔 0.30 eV），其余 2000 点均匀（步长≈0.100 eV）。
"""
import csv, json, os
import numpy as np

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_dornheim_fine")
_manifest = None

def _load_manifest():
    global _manifest
    if _manifest is None:
        with open(os.path.join(DB_DIR, "manifest.csv"), newline="") as f:
            _manifest = list(csv.DictReader(f))
    return _manifest

def grid_axes():
    man = _load_manifest()
    tes = sorted({float(r["Te_eV"]) for r in man})
    nes = sorted({float(r["ne_m3"]) for r in man})
    return tes, nes

def load_case(te_eV, ne_m3):
    for r in _load_manifest():
        if float(r["Te_eV"]) == te_eV and float(r["ne_m3"]) == ne_m3:
            z = np.load(os.path.join(DB_DIR, r["file"]))
            return {"E_eV": z["E_eV"], "S_free": z["S_free"], "S_tot": z["S_tot"],
                    "n_e": float(z["n_e"]), "k": float(z["k"]),
                    "meta": json.loads(str(z["meta_json"])), "file": r["file"]}
    raise KeyError(f"网格外: Te={te_eV} eV, ne={ne_m3} m^-3（用 grid_axes() 查网格）")

if __name__ == "__main__":
    tes, nes = grid_axes()
    print(f"Te: {tes[0]:.0f}..{tes[-1]:.0f} eV ({len(tes)} 点); ne: {nes[0]:.2e}..{nes[-1]:.2e} m^-3 ({len(nes)} 点)")
    d = load_case(11.0, 1.55e30)
    print("sample (11 eV, 1.55e30):", d["file"], "npts =", len(d["E_eV"]), "k =", d["k"])
