"""读取已有数据库：列格点、查看字段、画谱或导出 CSV，不重新计算。

用户配置在下方；命令行参数可以覆盖。
新 A 库 db_glz_a_sif：S_tot、S_elastic、S_inelastic，共用总谱峰值归一尺度。
新 B 库 db_glz_b_sif：S_free 为未展宽谱；比较实验应读 S_free_broadened。
2026-10-30更新：B库生成时先展宽再峰值归一，重算后的S_free_broadened峰值为1。
读取时不再次展宽、不重新归一化，也不把缺失格点替换为附近格点。

例：
  python reading.py --db db_glz_a_sif --list
  python reading.py --db db_glz_a_sif --te 12 --ne 3e29 --spectrum S_tot S_elastic S_inelastic
  python reading.py --db db_glz_b_sif --te 12 --ne 3e29 --spectrum S_free_broadened
  python reading.py --db db_glz_b_sif --te 12 --ne 3e29 --info
  python reading.py --db db_glz_a_sif --te 12 --ne 3e29 --loss-axis --export spectrum.csv --no-show
旧 db_v1/db_v2（--rho）和 db_dornheim_fine（--ne）仍兼容。
Python 接口：load_case('db_glz_a_sif', 12, ne=3e29)，返回数组及 meta 字典。
完整说明见 database_sif_reading_guide.md。
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

# ================= 用户配置（命令行参数可覆盖） =================
DATABASE = "db_glz_a_sif"  # 新库 db_glz_a_sif / db_glz_b_sif；也支持旧库
TEMPERATURES = None  # None：GLZ用(12,)，旧库用(10,30,50)，细网格用(5,10,20)
GLZ_NE_M3 = 3e29  # GLZ电子密度 [m^-3]；ne=1e29+j*5e28，j=0..38
NE_M3 = 1e30  # 原 db_dornheim_fine 默认电子密度 [m^-3]
RHO_GCC = 10.0  # 原 db_v1/db_v2 默认质量密度 [g/cm³]
THETA_DEG = None  # None：电子密度库沿用库内角度；旧质量密度库用160°
SPECTRUM = None  # None：A自动读S_tot，B自动读S_free_broadened
# 也可写 SPECTRUM = ("S_tot", "S_elastic", "S_inelastic")，仅用于含这些字段的库
LOSS_AXIS = False  # True：损失为正；False：原库光子位移，红移为负
# =============================================================
ROOT = Path(__file__).resolve().parent
SPECTRUM_FIELDS = ("S_tot", "S_elastic", "S_inelastic", "S_free_broadened",
                   "S_free", "S_elastic_fit")


def resolve_database(database):
    directory = Path(database).expanduser()
    return directory if directory.is_absolute() else ROOT/"database"/directory


def _close(a, b):
    return np.isclose(float(a), float(b), rtol=1e-10, atol=0)


def read_index(database):
    """旧库使用 manifest；GLZ 从实际 npz 的 meta_json 建索引。

    GLZ共享manifest是可重复追加的运行日志，不能当唯一格点表。
    索引只读元数据，不读取全库谱数组，也不重新运行物理引擎。
    """
    directory = resolve_database(database)
    manifest = directory/"manifest.csv"
    rows = []
    if manifest.exists():
        with manifest.open(encoding="utf-8-sig", newline="") as f:
            for source in csv.DictReader(f):
                filename = source.get("filename") or source.get("file")
                if source.get("status") not in ("ok", "done") or not filename:
                    continue
                key = "ne_m3" if source.get("ne_m3") else "rho_gcc"
                rows.append(dict(file=filename, Te_eV=float(source["Te_eV"]),
                    density_key=key, density=float(source[key]),
                    theta_deg=float(source["theta_deg"]) if source.get("theta_deg") else None))
    else:
        for path in sorted(directory.glob("case_*.npz")):
            with np.load(path, allow_pickle=False) as d:
                meta = json.loads(str(d["meta_json"]))
            density = meta.get("ne", meta.get("ne_m3"))
            key = "ne_m3"
            if density is None:
                key, density = "rho_gcc", meta.get("rho_gcc")
            if density is None:
                raise ValueError(f"{path.name} 的元数据缺少密度")
            rows.append(dict(file=path.name, Te_eV=float(meta["Te_eV"]),
                density_key=key, density=float(density), theta_deg=float(meta["theta_deg"])))
    if not rows:
        raise ValueError(f"库中没有已完成的有效算例：{directory}")
    if len({r["density_key"] for r in rows}) != 1:
        raise ValueError("数据库混用电子密度和质量密度索引")
    if any(not np.isfinite([r["Te_eV"], r["density"]]).all() for r in rows):
        raise ValueError("数据库索引中有非有限温度/密度")
    return rows


def load_case(database, te, *, ne=None, rho=None, theta=None, index=None):
    """精确选格点，返回按 E_eV 升序的数组及 meta/path；强度保持原样。

    ne单位m^-3，rho单位g/cm³。明确要求传ne或rho；默认画图参数只在main应用。
    """
    directory = resolve_database(database)
    rows = read_index(directory) if index is None else index
    electronic = rows[0]["density_key"] == "ne_m3"
    if electronic and (ne is None or rho is not None):
        raise ValueError("该库按电子密度索引，请传 ne=...，不要传 rho")
    if not electronic and (rho is None or ne is not None):
        raise ValueError("该库按质量密度索引，请传 rho=...，不要传 ne")
    density = ne if electronic else rho
    matches = [r for r in rows if _close(r["Te_eV"], te) and _close(r["density"], density)
        and (theta is None or r["theta_deg"] is None or _close(r["theta_deg"], theta))]
    if len(matches) != 1:
        raise ValueError(f"未找到唯一算例：Te={te}, {rows[0]['density_key']}={density}, "
                         f"theta={theta}；用 --list 查看实际格点")
    path = directory/matches[0]["file"]
    with np.load(path, allow_pickle=False) as d:
        result = {key: d[key].copy() for key in d.files if key != "meta_json"}
        meta = json.loads(str(d["meta_json"]))
    actual_density = meta.get("ne", meta.get("ne_m3")) if electronic else meta.get("rho_gcc")
    if not _close(meta["Te_eV"], te) or actual_density is None or not _close(actual_density, density):
        raise ValueError(f"{path.name} 的温度/密度与索引不一致")
    if theta is not None and not _close(meta["theta_deg"], theta):
        raise ValueError(f"该库固定角度为 {meta['theta_deg']}°，与请求 {theta}° 不一致")
    energy = result["E_eV"]
    if energy.ndim != 1 or not len(energy) or not np.isfinite(energy).all():
        raise ValueError(f"{path.name} 的能量网格无效")
    order = np.argsort(energy)
    if np.any(np.diff(energy[order]) <= 0):
        raise ValueError(f"{path.name} 存在重复能量坐标")
    result["E_eV"] = energy[order]
    for key in (*SPECTRUM_FIELDS, "rms_valid_mask"):
        if key in result:
            if result[key].shape != energy.shape or not np.isfinite(result[key]).all():
                raise ValueError(f"{path.name}:{key} 形状或数值无效")
            result[key] = result[key][order]
    result.update(meta=meta, path=path.resolve())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=DATABASE, help="database内库名，或数据库绝对路径")
    parser.add_argument("--te", nargs="+", type=float, default=TEMPERATURES)
    parser.add_argument("--rho", type=float, help="质量密度[g/cm³]")
    parser.add_argument("--ne", type=float, help="电子密度[m^-3]")
    parser.add_argument("--theta", type=float, default=THETA_DEG)
    default_spectra = [SPECTRUM] if isinstance(SPECTRUM, str) else SPECTRUM
    parser.add_argument("--spectrum", nargs="+", choices=SPECTRUM_FIELDS, default=default_spectra)
    parser.add_argument("--list", action="store_true", help="仅显示实际格点与可用谱字段")
    parser.add_argument("--info", action="store_true", help="查看字段、峰值和元数据，不绘图")
    parser.add_argument("--loss-axis", action="store_true", default=LOSS_AXIS, help="横轴改为损失为正")
    parser.add_argument("--valid-only", action="store_true", help="仅画/导出rms_valid_mask为True的点")
    parser.add_argument("--export", type=Path, help="CSV导出路径，不改库内强度尺度")
    parser.add_argument("--save", type=Path, help="图片保存路径")
    parser.add_argument("--no-show", action="store_true", help="不弹出窗口")
    args = parser.parse_args(argv)
    try:
        directory = resolve_database(args.db)
        rows = read_index(directory)
        electronic = rows[0]["density_key"] == "ne_m3"
        glz = directory.name.startswith("db_glz_")
        print("数据库：", directory.resolve())
        print("实际文件/索引数量：", len(rows))
        print("温度 Te [eV]：", sorted({r["Te_eV"] for r in rows}))
        print(f"密度 {rows[0]['density_key']}：", sorted({r["density"] for r in rows}))
        if args.list:
            angles = {r["theta_deg"] for r in rows if r["theta_deg"] is not None}
            with np.load(directory/rows[0]["file"], allow_pickle=False) as d:
                print("首个算例可用谱字段：", [key for key in SPECTRUM_FIELDS if key in d.files])
                if not angles:
                    angles.add(float(json.loads(str(d["meta_json"]))["theta_deg"]))
            print("角度[deg]：", sorted(angles))
            print("E_eV为光子位移，损失为负；--loss-axis可改为损失为正")
            return
        if electronic and args.rho is not None:
            raise ValueError("该库按电子密度索引，请使用 --ne，不要使用 --rho")
        if not electronic and args.ne is not None:
            raise ValueError("该库按质量密度索引，请使用 --rho，不要使用 --ne")
        density = ((args.ne if args.ne is not None else (GLZ_NE_M3 if glz else NE_M3))
            if electronic else (args.rho if args.rho is not None else RHO_GCC))
        theta = args.theta if args.theta is not None else (None if electronic else 160.0)
        temperatures = args.te if args.te is not None else ((12,) if glz else
            ((5, 10, 20) if electronic else (10, 30, 50)))
        cases, geometry, spectra = [], None, args.spectrum
        for te in temperatures:
            case = load_case(directory, te, ne=density if electronic else None,
                rho=None if electronic else density, theta=theta, index=rows)
            meta = case["meta"]
            actual = (float(meta["theta_deg"]), float(meta["E0_keV"]))
            if geometry is not None and geometry != actual:
                raise ValueError("选中的算例角度或探针能量不一致")
            geometry = actual
            if spectra is None:
                spectra = [next(k for k in ("S_tot", "S_free_broadened", "S_free", "S_inelastic") if k in case)]
            missing = [key for key in spectra if key not in case]
            if missing:
                available = [key for key in SPECTRUM_FIELDS if key in case]
                raise ValueError(f"{case['path'].name} 没有 {missing}；可用字段：{available}")
            print("读取：", case["path"])
            print("谱字段：", ", ".join(spectra), "；不重新归一化、不再次展宽")
            if args.info:
                print("字段：", [k for k in case if k not in ("meta", "path")])
                print("能量范围/点数：", case["E_eV"][[0, -1]].tolist(), len(case["E_eV"]))
                for key in [k for k in SPECTRUM_FIELDS if k in case]:
                    print(f"  {key}: min={case[key].min():.8g}, max={case[key].max():.8g}")
                print("meta_json：", json.dumps(meta, ensure_ascii=False, indent=2))
            energy = -case["E_eV"] if args.loss_axis else case["E_eV"]
            mask = (case["rms_valid_mask"].astype(bool) if "rms_valid_mask" in case
                    else np.full(len(energy), None, dtype=object))
            if args.valid_only and "rms_valid_mask" not in case:
                raise ValueError("该库没有rms_valid_mask，不能推断有效RMS区间")
            order = np.argsort(energy)
            if args.valid_only:
                order = order[mask[order]]
                if not len(order):
                    raise ValueError("该算例没有有效RMS点")
            cases.append((te, energy[order], {k: case[k][order] for k in spectra}, mask[order]))
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            with args.export.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Te_eV", rows[0]["density_key"], "theta_deg",
                    "energy_loss_eV" if args.loss_axis else "photon_shift_eV", *spectra, "rms_valid_mask"])
                for te, energy, values, mask in cases:
                    for i, value in enumerate(energy):
                        writer.writerow([te, density, geometry[0], value,
                            *(values[k][i] for k in spectra),
                            int(mask[i]) if mask[i] is not None else ""])
            print("CSV已保存：", args.export.resolve())
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.error(str(error))
    if args.info:
        return
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for te, energy, values, mask in cases:
        for key in spectra:
            ax.plot(energy, values[key], label=f"Te={te:g} eV, {key}")
    density_label = f"ne={density:.3g} m^-3" if electronic else f"rho={density:g} g/cm³"
    ax.set(xlabel="Energy loss (eV)" if args.loss_axis else "Photon energy shift (eV)",
        ylabel="Stored intensity (no renormalization)",
        title=f"{directory.name}: {density_label}, θ={geometry[0]:g}°, E0={geometry[1]:g} keV")
    ax.legend()
    fig.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print("图片已保存：", args.save.resolve())
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
