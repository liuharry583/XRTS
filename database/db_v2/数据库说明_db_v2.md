# 数据库说明 — db_v2（RPA 理论数据库 · 对齐 wsb 交付设定版）

构建日期：2026-10-25 晚 ｜ 构建脚本：`RPA数据库_批量跑批_v2.py` ｜ 构建耗时：1024 s
与 db_v1 的关系：**网格、能量窗、E0、材料全部相同，只改物理开关**——把配置对齐到
wsb 在任务 2 中实际交付的 RMS 反演设定（其两个 run 脚本均为 NU_MODE="ZERO"），
使查库结果与他的流水线现算结果可逐位对照。

```python
class DbV2:
    """与 DbV1 同网格；物理开关对齐 wsb 交付的 RMS 配置。"""

    # —— 材料与探针（构造参数）——
    material: str        = "Be"        # 纯铍
    ZA: float            = 4.0
    Zf: float            = 2.0         # 常数
    E0_keV: float        = 5.0

    # —— 物理开关（与 v1 的差异只有这三行）——
    NU_MODE: str         = "const"
    NU_FRAC: float       = 0.0         # ★ 碰撞关闭 ν=0（v1 为 0.05）
    SII_MODE: str        = "arkhipov"  # ★ S_ii 换 Arkhipov 双组分模型（v1 为 Debye–Hückel）
    ELASTIC_WIDTH_eV: float = 2.0      # 占位，同 v1
    METHOD: str          = "auto"      # 同 v1

    # —— 网格（同 v1）——
    rho_grid: Grid       = Grid(0, 50, step=5, unit="g/cm^3")    # 11 点
    Te_grid: Grid        = Grid(0, 50, step=5, unit="eV")        # 11 点
    theta_grid: Grid     = Grid(0, 170, step=10, unit="deg")     # 18 点
    energy_window: tuple = (-400.0, 400.0)   # eV
    nw: int              = 600

    # —— 引擎 ——
    engine: str          = "draft3.py（带 NU_MODE/SII_MODE 双开关、BMA 符号已修复的版本；批量脚本逐 case 打补丁）"
```

## 网格与 case 统计

与 db_v1 完全相同：manifest 2178 行 = 实算 1700 + 零填充 187（ρ=0）+ 跳过 291
（θ=0 / Te=0 奇点）；npz 1887 个，目录 38 MB。ne 换算与范围同 v1
（约 6.68×10²⁹ – 6.68×10³⁰ m⁻³）。

## 方法

- 物理模型与 v1 相同（Lindhard χ₀ → RPA → Mermin → FDT；Chihara 离子项），
  仅两处模型替换：碰撞频率 ν 由 0.05ωp 改为 0；S_ii 由 Debye–Hückel 改为
  Arkhipov 双组分模型（出自张小丁论文公式 1.7，含量子修正有效温度）。
- 注：引擎虽已移植 BMA 动态碰撞频率并修复其符号错误（参考实现整体差一个负号），
  但本库按“对齐交付设定”原则未启用 BMA（NU_MODE="const"、NU_FRAC=0）。

## 验证

- QC：0 FAIL（1015 条 WARN，同 v1 的判据伪影类）。
- 抽查 case（ρ=10、Te=20、θ=170°）与 wsb 原版 physics_core（ν=0 + Arkhipov、
  lam0 手工修正）在同一参数下 **S_free、S_tot 逐位一致**——两引擎同源就此实测坐实。
- 全部 1700 个实算 case 的 k 与 E0=5 keV 公式逐个相符（lam0 无污染，同 v1）。

## 产物与读取

- 目录 `db_v2/`：npz 字段与 v1 相同（`E_eV` 600 点降序、`S_free`、`S_tot`、`n_e`、
  `k`、`mu_eV`、`wp`、`nu`、`meta_json`），`manifest.csv` 同格式。
- 打包：`db_v2.zip`（19 MB，包根读取脚本指向 db_v2）。

## 备注

- 本版定位：与任务 2（RMS 反演）对接的对照库。做 Dornheim 拟合时它仍有先天不足
  （E0=5 keV 与实验 2.96 keV 不符、弹性宽为占位 2 eV、Te/ρ 步长 5 太粗、能量采样
  1.34 eV/点太稀、低 ne 区缺失）——这些由 db_dornheim_fine 针对性解决。
