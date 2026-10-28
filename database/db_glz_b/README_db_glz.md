# db_glz 双库说明（Glenzer 口径，2026-10-26 建）

组会（2026-10-26，梁泂航）任务2 的理论数据库：为与 Glenzer et al. 2007
（PRL 98, 065002）Fig. 2 数字化数据的 RMS 对比而建。引擎为 draft.py
（整谱同一高斯仪器展宽，即任务1 的 Glenzer 式 (54) 修正已包含在内）。

## 规格

```
固定口径（全库统一，源自 Glenzer 2007 原文）:
  E0 = 2.96 keV（Cl Ly-alpha）, theta = 40 deg, 材料 Be（ZA = 4）
  Zf = 2.2（原文实测电离度）
  ELECTRON_METHOD = "integral"（有限温 Lindhard，含简并）
  NU_MODE = "const", nu_rad_s = 5.0e14 s^-1（原文 nu_ei，Born 近似 + Mermin）
  SII_MODE = "dh"
  INSTRUMENT_FWHM_eV = 8.7（数字化 SIF 实测 FWHM，高斯形）

网格:
  Te = 1..31 eV，步长 1（31 个）
  ne = 1e29 + j*5e28 m^-3, j = 0..38（39 个）
  共 1209 case，全部计算成功（manifest status 全 ok）

能量轴:
  E_eV 为引擎约定（光子能量位移，下移为负），降序，窗 -70..+50 eV，步长 0.1 eV
  名义 1200 点；465 个 case 为 1199 点——引擎为避开 omega=0 处 FDT 的 0/0
  掩去了零频最近的 1 个点（落在弹性峰中心，对使用无影响），属引擎固有行为
```

## 两库定义（归一化口径经 Harry 2026-10-26 确认）

- **库A（db_glz_a/，含 Sii）**：每 case 存 `S_tot`（含弹性/离子项的整谱，
  已统一仪器展宽），按**全谱最大值归一为 1**（引擎 normalize=True）。
- **库B（db_glz_b/，不含 Sii）**：每 case 存 `S_free`（纯自由电子谱），
  按**自身最大值归一为 1**。注意：引擎返回的 S_free 是**未展宽**谱，
  与实验非弹性部分比较前需在比较端做展宽/卷积（与 wsb 管线
  「理论谱在拟合端卷数字化 SIF」的用法一致）。

每 case 一个 .npz（文件名 case_T{Te:02d}_N{j:02d}.npz，两库同名）：
`E_eV`、谱数组（`S_tot` 或 `S_free`）、`meta_json`（完整口径记录）。
清单 db_glz_manifest.csv；读取器 读取数据库_glz.py
（load_case_a / load_case_b 按 (Te, ne) 精确匹配格点）。

## 使用注意

- **低温区峰位**：Te ≤ 6 eV 的 167 个 case 中，S_tot 的最大值出现在
  plasmon 处而非弹性峰（强简并下 plasmon 阻尼弱、峰高，弹性峰相对被
  压低；弹性峰仍在，相对高度 0.1–0.7）。归一化定义不变（全谱峰 = 1），
  但若按「弹性峰 = 1」的预期使用这批 case，需自行留意。
- 横轴约定：与论文图（损失为正）比较时整体取 -E。
- 本库只含理论谱；实验侧的 Sii 扣除与 RMS 计算不在本库范围内。

## QC

- 1209/1209 ok，0 FAIL；两库每 case 最大值精确为 1.000000。
- 抽查 (Te=12, ne=3e29) 独立复算：库A、库B 与引擎现算结果逐位一致。
- 同 case 的库B 驼峰位于 -29.07 eV（引擎约定），与 Glenzer 图
  （展宽后损失 +27 eV）对应。
