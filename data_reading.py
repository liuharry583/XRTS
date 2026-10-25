"""读取 RPA 理论数据库 db_v1（与 RPA数据库_批量跑批.py 配套）。

每个 case 是一个 .npz 文件（numpy 压缩存档），里面有：
  E_eV     (600,)  光子能量位移网格 [eV]，负 = 能量下移（论文惯例）
  S_free   (600,)  自由电子项 S^0_ee(k,ω)，单电子归一，绝对单位
  S_tot    (600,)  总谱 = Zf * S_free + 离子/弹性项（含弹性峰）
  k        标量    散射波矢 [m^-1]
  n_e      标量    电子数密度 [m^-3]
  mu_eV    标量    化学势 [eV]
  wp       标量    等离子体频率 [rad/s]
  nu       标量    碰撞频率 [rad/s]
  meta_json 字符串  全部输入参数的 JSON（rho/Te/theta/E0/Zf/材料/方法…）

manifest.csv 是全库索引：一行一个 case（rho, Te, theta, 文件名, 状态…），
用它先查有哪些 case，再按需 load 对应的 npz，不要一次性全读进内存。

用法示例见文件末尾 __main__。
"""
import csv
import json
import os

import numpy as np

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_v1")


def case_filename(rho, te, th):
    """网格坐标 → 文件名（与跑批脚本的命名规则一致）"""
    return f"case_r{int(rho):02d}_T{int(te):02d}_th{int(th):03d}.npz"


def load_case(rho, te, th, db_dir=DB_DIR):
    """按 (密度 g/cm³, 温度 eV, 散射角 °) 读一个 case，返回普通 dict。

    跳过的 case（theta=0 / Te=0）没有文件，会抛 FileNotFoundError；
    rho=0 的 case 是真空零填充，S 全 0，meta['note'] 里有标注。
    """
    path = os.path.join(db_dir, case_filename(rho, te, th))
    d = np.load(path, allow_pickle=False)
    return {
        "E": d["E_eV"],                # 能量位移 [eV]
        "S_free": d["S_free"],         # 自由电子项
        "S_tot": d["S_tot"],           # 总谱（含弹性峰）
        "k": float(d["k"]),            # 波矢 [m^-1]
        "n_e": float(d["n_e"]),        # 电子数密度 [m^-3]
        "mu_eV": float(d["mu_eV"]),   # 化学势 [eV]
        "wp": float(d["wp"]),          # 等离子体频率 [rad/s]
        "nu": float(d["nu"]),          # 碰撞频率 [rad/s]
        "meta": json.loads(str(d["meta_json"])),  # 全部输入参数
        "path": path,
    }


def load_manifest(db_dir=DB_DIR):
    """读全库索引，返回 list[dict]，每行一个 case（含 status 字段）。"""
    with open(os.path.join(db_dir, "manifest.csv"), newline="") as f:
        return list(csv.DictReader(f))


def iter_cases(db_dir=DB_DIR, status=None):
    """遍历全库并逐个 yield case dict（status=None 不过滤；
    status='ok' 只给真正算了的，'zero_fill' 给零填充的）。"""
    for row in load_manifest(db_dir):
        if status is not None and row["status"] != status:
            continue
        yield load_case(int(row["rho_gcc"]), int(row["Te_eV"]),
                        int(row["theta_deg"]), db_dir)


if __name__ == "__main__":
    # --- 演示 1：全库概况 ---
    man = load_manifest()
    from collections import Counter
    print(f"全库 {len(man)} 个 case，状态分布：{dict(Counter(r['status'] for r in man))}")

    # --- 演示 2：读一个 case 看结构 ---
    c = load_case(rho=10, te=20, th=90)
    print(f"\ncase rho=10 g/cm³, Te=20 eV, θ=90°:")
    print(f"  能量网格 {c['E'][0]:.0f}..{c['E'][-1]:.0f} eV，共 {len(c['E'])} 点")
    print(f"  k={c['k']:.3e} /m, n_e={c['n_e']:.3e} /m³, mu={c['mu_eV']:.1f} eV")
    print(f"  S_tot 峰值 {c['S_tot'].max():.3e}，位于 {c['E'][np.argmax(c['S_tot'])]:.0f} eV（弹性峰）")

    # --- 演示 3：画三条谱线存图 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    for te in (10, 30, 50):
        cc = load_case(rho=10, te=te, th=160)
        plt.plot(cc["E"], cc["S_tot"], label=f"Te={te} eV")
    plt.xlabel("Photon energy shift (eV)")
    plt.ylabel("S(k,ω) (arb. units)")
    plt.title("db_v1 demo: rho=10 g/cm³, θ=160°")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_v1_读取演示.png")
    plt.savefig(out, dpi=110)
    print(f"\n演示图已存：{out}")
