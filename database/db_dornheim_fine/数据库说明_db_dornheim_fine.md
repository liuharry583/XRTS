# 数据库说明 — db_dornheim_fine（Dornheim 拟合专用细网格库）

构建日期：2026-10-26 ｜ 构建脚本：`RPA数据库_批量跑批_dornheim_fine.py` ｜ 构建耗时：96 s
定位：不是通用库，而是为 Dornheim 2022（Nat Commun 13, 7911）Fig 2a 那发
Glenzer 温密 Be 实验的 RMS 反演**量身固定全部实验条件**的切面细库。
网格方案来自 Codex 2026-10-26 的两级扫描建议（粗/细两级取并集，一次建全）。

```python
class DbDornheimFine:
    """Dornheim 拟合专用：实验条件全固定，只扫 (Te, ne)。"""

    # —— 材料与实验条件（构造参数，全部对齐 Dornheim Fig 2a 实验）——
    material: str        = "Be"        # 纯铍
    ZA: float            = 4.0
    Zf: float            = 2.0
    E0_keV: float        = 2.96        # ★ 实验探针能量（图内印字，非四舍五入）
    theta_deg: float     = 40.0        # ★ 散射角固定

    # —— 物理开关（与 wsb 交付的 RMS 设定逐项一致）——
    NU_MODE: str         = "const"
    NU_FRAC: float       = 0.0         # 碰撞关闭 ν=0
    SII_MODE: str        = "arkhipov"  # Arkhipov 双组分 S_ii
    ELASTIC_WIDTH_eV: float = 18.0     # ★ 弹性宽取该实验的实际展宽量级（非占位）
    METHOD: str          = "maxwell"   # ★ 强制 Maxwell 解析分支（其流水线为提速的既定约定）

    # —— 网格（以 ne 为拟合坐标）——
    Te_grid: Grid        = Grid(1, 31, step=1, unit="eV")              # 31 点
    ne_grid: Grid        = Grid(1e29, 2e30, step=5e28, unit="m^-3")    # 39 点
    energy_window: tuple = (-100.0, 100.0)   # eV
    nw: int              = 2000              # 步长 200/1999 ≈ 0.100 eV，原生密采样

    # —— 引擎 ——
    engine: str          = "draft3.py 重构版（PhysicsConfig 配方卡接口，逐 case 传 config，物理公式与旧版逐位一致）"
```

## 网格与 case 统计

- 总计 **1209 case（31×39），全部实算**——网格不含 ρ=0 / Te=0 / θ=0，无零填充、无跳过。
- manifest 每行附 `rho_equiv_gcc`：按 Be、Zf=2 换算的等效质量密度
  （ρ = ne/Zf · A·u），仅作对照，拟合坐标以 ne 为准。
- 文件名 `case_T{Te:02d}_N{j:02d}.npz`（j 为 ne 网格序号）；目录 52 MB。

## 方法

- 物理模型与 v1/v2 相同（Lindhard χ₀ → RPA → Mermin → FDT；Chihara 离子项），
  本库固定走 Maxwell 解析分支：每条谱只剩 ρ_cloud 需要一次静态 ε(k,0) 积分，
  其余为解析求值——这是本库单 case 仅 0.079 s（对比 v2 积分分支约 0.6 s）的原因。
- 参数传递用新版引擎的 PhysicsConfig：每个 case 一张不可变“配方卡”，派生量
  （T、λ₀、ωp、κ、ν、μ）由引擎按卡现推——从结构上杜绝 lam0 未更新类错误。

## 产物与读取

- 目录 `db_dornheim_fine/`：每 npz 含 `E_eV`（降序 100→−100）、`S_free`、`S_tot`、
  `n_e`、`k`、`meta_json`（完整记录本 case 的构造参数）；`manifest.csv` 含
  (i_Te, j_ne, Te_eV, ne_m3, ne_cm3, rho_equiv_gcc, file, status)。
- 打包：`db_dornheim_fine.zip`（50 MB，包根含 `读取数据库_dornheim_fine.py`
  与 `README_db_dornheim_fine.md`）。读取：`load_case(te_eV, ne_m3)` 须命中网格点。

## 备注与已知问题

1. **124 个 case 是 1998 点而非 2000 点**：ne ≥ 1.85×10³⁰ 时，引擎避开 ω=0 处
   FDT 0/0 的掩膜（|ω| ≤ 1e-3·ωp）盖住中心 2 个网格点，在 E≈0 留下 0.30 eV 间隔。
   引擎固有行为（v1/v2 同样如此），位于弹性峰内部、远离拟合窗，插值/卷积不受影响。
2. **ν=0 时 RMS 谷底为采样伪影（重要）**：Maxwell 分支在 ν=0 时 plasmon 峰是
   FWHM≈0.01 eV 的针尖，采样面积随网格对齐混沌涨落并经 SIF 卷积远尾传进拟合窗。
   本库重扫的表观最小（Te=12 eV、ne=1.6×10³⁰ m⁻³、RMS=0.0176；全场中位 0.0357）
   与此前 45×45 扫描的 (10.89 eV, 0.0175) 同为抽奖样本，不可作物理结论；
   三项实证与处理建议见包内 README。加密网格解决不了此问题，须先定碰撞约定
   或对针尖做面积守恒处理。
3. 前身：`db_dornheim` 切面库（2026-10-26 凌晨）用 wsb 扫描网格原样
   （Te linspace(1,30,45) × ne linspace(5e29, 2e30, 45)、窗 −55..+10 eV/800 点，
   其余设定同本库），用途是验证“查库 ≡ 现算”的闭环（已逐位对齐）；本库是其加密继任者。
