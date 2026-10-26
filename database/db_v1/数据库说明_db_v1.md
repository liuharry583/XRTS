# 数据库说明 — db_v1（RPA 理论数据库 · 初版）

构建日期：2026-10-25 ｜ 构建脚本：`RPA数据库_批量跑批.py`
依据规格：《RPA理论数据库_构建规格.md》（2026-10-19 组会版），XRTS 任务 1 首版。

```python
class DbV1:
    """通用 RPA 理论谱库：(ρ, Te, θ) 三维网格，组内 v1 约定。"""

    # —— 材料与探针（构造参数）——
    material: str        = "Be"        # 纯铍
    ZA: float            = 4.0         # 原子序数
    Zf: float            = 2.0         # 自由电子电离度，常数（zf_of_rho_T 恒定取值）
    E0_keV: float        = 5.0         # 探针光子能量

    # —— 物理开关 ——
    NU_MODE: str         = "const"
    NU_FRAC: float       = 0.05        # 碰撞频率 ν = 0.05 · ωp
    SII_MODEL: str       = "debye_huckel"   # S_ii 用 Debye–Hückel（引擎当时唯一实现）
    ELASTIC_WIDTH_eV: float = 2.0      # 弹性峰高斯宽，占位值（真实展宽由 RMS 阶段 SIF 卷积给出）
    METHOD: str          = "auto"      # 引擎默认：集体区走积分分支、非集体区走 Maxwell

    # —— 网格 ——
    rho_grid: Grid       = Grid(0, 50, step=5, unit="g/cm^3")    # 11 点
    Te_grid: Grid        = Grid(0, 50, step=5, unit="eV")        # 11 点
    theta_grid: Grid     = Grid(0, 170, step=10, unit="deg")     # 18 点
    energy_window: tuple = (-400.0, 400.0)   # eV
    nw: int              = 600               # 每条谱采样点数，步长 ≈1.34 eV

    # —— 引擎 ——
    engine: str          = "Sii.py（draft3 前身；importlib 动态载入，setup_case 逐 case 打补丁改全局量）"
```

## 网格与 case 统计

| 状态 | 数量 | 说明 |
|---|---|---|
| done（实算） | 1700 | ρ>0、Te>0、θ>0，即 10×10×17 |
| zero_fill（零填充） | 187 | ρ=0 且 θ>0：S 全 0，不调用引擎（含 Te=0 行） |
| skipped_singular（跳过） | 291 | θ=0 全部 121（k=0 奇点）+ Te=0 且 ρ>0、θ>0 的 170（chem_pot 在 T=0 奇异） |
| **合计（manifest 行）** | **2178** | npz 文件 1887 个（实算 + 零填充），目录 38 MB |

ne 由 ρ 换算：ne = Zf · ρ / (A_Be · u)，网格内 ne 范围约 6.68×10²⁹ – 6.68×10³⁰ m⁻³。

## 方法

- 物理模型：有限温 Lindhard χ₀（自适应 η）→ RPA 介电函数 → Mermin（含常数碰撞频率 ν）
  → 精确 FDT 前因子得 S_free；离子项用 Chihara 分解 |f_i + ρ_cloud|² · S_ii(DH) · 高斯弹性峰。
- 每个 case 在跑批脚本的 `setup_case` 中按 (ρ, Te, θ) 显式重算全部派生量
  （T、λ₀、ωp、κ、ν、μ）后再调用引擎 `compute()`——λ₀ 逐 case 重算，数据库不带
  wsb 脚本那种 lam0 未更新问题（已全量核查：1700 个实算 case 存的 k 与 5 keV 公式逐个相符）。
- 低 Te 行按决定单方案 (a) 全量实算（不外推、不留空）。

## 产物与读取

- 目录 `db_v1/`：每 case 一个 `case_r{ρ}_T{Te}_th{θ}.npz`，含 `E_eV`（600 点，降序
  400→−400）、`S_free`、`S_tot`、`n_e`、`k`、`mu_eV`、`wp`、`nu`、`meta_json`；另有 `manifest.csv`。
- 打包：`db_v1.zip`（19 MB，读取脚本 `读取数据库.py` 位于包根）。
- QC：0 FAIL；约 1004 条 WARN 全部来自 QC 判据本身在简并区的预期失效（经典 Debye
  求和规则、反冲峰位锚点），逐项核过谱本身无误。

## 备注

- 参数来源：E0/材料锚定 Dornheim 的 Be 实验（5 keV 由其 ~4.75 keV 取整）；
  Zf=2.0、ν=0.05ωp、弹性宽 2 eV 为组内 v1 约定；Kritcher 2011 只贡献 RMS 反演方法，未贡献任何参数。
- 本版已被 db_v2（同网格、对齐 wsb 交付设定）取代为对接 RMS 的版本；v1 保留作
  “组内约定版”对照。
