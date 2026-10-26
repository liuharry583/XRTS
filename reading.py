"""从 database/<库名>/manifest.csv 选取算例并画谱，不重新计算。

直接运行使用下面的配置；也可运行：
python reading.py --db db_v2 --te 10 30 50 --rho 10 --theta 160
python reading.py --db db_dornheim_fine --te 5 10 20 --ne 1e30
python reading.py --db db_dornheim_fine --list
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# ================= 用户配置（命令行参数可覆盖） =================
DATABASE = "db_dornheim_fine"  # 可选 db_v1 / db_v2 / db_dornheim_fine
TEMPERATURES = None  # None：旧库(10,30,50)，细网格库(5,10,20)；也可填元组
RHO_GCC = 10.0  # db_v1/db_v2 使用质量密度 [g/cm³]
NE_M3 = 1e30  # db_dornheim_fine 使用电子密度 [m^-3]
THETA_DEG = None  # None：旧库160°；细网格库读取其固定角度
SPECTRUM = "S_tot"  # S_tot 或 S_free
# =============================================================
ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default=DATABASE, help="database 内的库名，或数据库绝对路径"
    )
    parser.add_argument("--te", nargs="+", type=float, default=TEMPERATURES)
    parser.add_argument("--rho", type=float, help="旧库质量密度 [g/cm³]")
    parser.add_argument("--ne", type=float, help="细网格库电子密度 [m^-3]")
    parser.add_argument("--theta", type=float, default=THETA_DEG)
    parser.add_argument("--spectrum", choices=["S_tot", "S_free"], default=SPECTRUM)
    parser.add_argument("--list", action="store_true", help="仅显示可用网格")
    parser.add_argument("--save", type=Path, help="图片保存路径")
    parser.add_argument("--no-show", action="store_true", help="不弹出窗口")
    args = parser.parse_args(argv)
    directory = Path(args.db).expanduser()
    if not directory.is_absolute():
        directory = ROOT / "database" / directory
    try:
        with (directory / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = [
                r
                for r in csv.DictReader(f)
                if r.get("status") in ("ok", "done")
                and (r.get("filename") or r.get("file"))
            ]
        if not rows:
            raise ValueError("库中没有已完成的有效算例")
        fine = "ne_m3" in rows[0]
        density_key = "ne_m3" if fine else "rho_gcc"
        print("数据库：", directory.resolve())
        print("温度 Te [eV]：", sorted({float(r["Te_eV"]) for r in rows}))
        print(f"密度 {density_key}：", sorted({float(r[density_key]) for r in rows}))
        if args.list:
            if "theta_deg" in rows[0]:
                print("角度 [deg]：", sorted({float(r["theta_deg"]) for r in rows}))
            else:
                with np.load(directory / rows[0]["file"], allow_pickle=False) as d:
                    print(
                        "固定角度 [deg]：", json.loads(str(d["meta_json"]))["theta_deg"]
                    )
            return
        if fine and args.rho is not None:
            raise ValueError("该库按电子密度索引，请使用 --ne，不要使用 --rho")
        if not fine and args.ne is not None:
            raise ValueError("该库按质量密度索引，请使用 --rho，不要使用 --ne")
        density = (
            (NE_M3 if args.ne is None else args.ne)
            if fine
            else (RHO_GCC if args.rho is None else args.rho)
        )
        theta = args.theta if args.theta is not None else (None if fine else 160.0)
        temperatures = (
            args.te if args.te is not None else ((5, 10, 20) if fine else (10, 30, 50))
        )
        curves = []
        geometry = None
        for te in temperatures:
            matches = [
                r
                for r in rows
                if np.isclose(float(r["Te_eV"]), te, rtol=1e-10, atol=0)
                and np.isclose(float(r[density_key]), density, rtol=1e-10, atol=0)
                and (
                    fine or np.isclose(float(r["theta_deg"]), theta, rtol=1e-10, atol=0)
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"未找到唯一算例：Te={te}, {density_key}={density}, theta={theta}；用 --list 查看网格"
                )
            filename = matches[0].get("filename") or matches[0]["file"]
            with np.load(directory / filename, allow_pickle=False) as d:
                meta = json.loads(str(d["meta_json"]))
                actual = (float(meta["theta_deg"]), float(meta["E0_keV"]))
                if theta is not None and not np.isclose(
                    theta, actual[0], rtol=1e-10, atol=0
                ):
                    raise ValueError(
                        f"该库固定角度为 {actual[0]}°，与请求的 {theta}° 不一致"
                    )
                if geometry is not None and geometry != actual:
                    raise ValueError("选中的算例角度或探针能量不一致")
                geometry = actual
                order = np.argsort(d["E_eV"])
                curves.append((te, d["E_eV"][order], d[args.spectrum][order]))
            print("读取：", filename)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for te, energy, spectrum in curves:
        ax.plot(energy, spectrum, label=f"Te={te:g} eV")
    density_label = f"ne={density:.3g} m^-3" if fine else f"rho={density:g} g/cm³"
    ax.set(
        xlabel="Photon energy shift (eV)",
        ylabel=f"{args.spectrum} (arb. units)",
        title=f"{directory.name}: {density_label}, θ={geometry[0]:g}°, E0={geometry[1]:g} keV",
    )
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
