# 三组实验谱与实测 SIF 数据库的 RMS 等值线

本说明对应 `rms_sif_contours.py`。2026-10-31 已用当前文件夹的数据完成三组全网格计算。
脚本独立读取 NumPy 数据库，不依赖 `reading.py` 或物理引擎；不会重算、重新展宽或改写输入数据库与实验文件。

## 1. 交付内容与实验文件对应

| 组名 / 输出前缀 | 实验文件 | 理论数据库 / 字段 | 比较的强度尺度 |
|---|---|---|---|
| `total` | `exp_elastic_1_negative.txt` | `database/db_glz_a_sif` / `S_tot` | 总谱峰值归一，包含弹性与非弹性 |
| `inelastic_total` | `exp_noelastic_1_negative.txt` | `database/db_glz_a_sif` / `S_inelastic` | 沿用总谱归一尺度，已去弹性 |
| `inelastic_peak` | `exp_noelastic_2_negative.txt` | `database/db_glz_b_sif` / `S_free_broadened` | 非弹性谱按自身峰值归一 |

需求中写的 `exp_inelastic_1/2_negative.txt` 在当前目录实际名为上表的 `exp_noelastic_1/2_negative.txt`。
检查确认第二份非弹性实验数据约等于第一份乘以 `1.643068084885144`，两者的横轴完全一致，符合两种归一化尺度。
文件名带 `elastic` 的第一份在这里是**包含弹性峰的总谱**，不是单独弹性分量。

A 的 `S_inelastic` 已包含 Zf 权重，并与 `S_tot` 共用总谱归一因子；不能再次把它独立归一为 1。
B 必须读已经过 SIF 展宽且展宽后自身峰值归一为 1 的 `S_free_broadened`；不能读未展宽的 `S_free`。
脚本检查全库的归一化与 A 分量恒等式，不自动修补旧版 B 库的尺度。

## 2. 与参考论文的关系

参考：A. L. Kritcher 等，*In-Flight Measurements of Capsule Shell Adiabats in Laser-Driven Implosions*，
Physical Review Letters **107**, 015002 (2011)，[DOI: 10.1103/PhysRevLett.107.015002](https://doi.org/10.1103/PhysRevLett.107.015002)。

论文第 3 页 Fig. 3 上图用温度作横轴、电子密度作纵轴，将理论与实验的 RMS 绘为等值线，暗色区域表示较好的匹配。
正文及图注描述的是 Compton 非弹性特征的拟合，并讨论红翼对温度和密度的灵敏度。
文章没有明确给出该图采用的 RMS 数学定义、精确能区、逐点权重及强度处理细节。
因此本程序仿照它的**参数扫描与等值线呈现方式**，以下计算约定由本程序明确指定，不能声称完全复现作者的数值处理。

论文实验使用约 9 keV 探针和 135° 散射；当前双库是 2.96 keV、40°。
本程序对比的是你的实验文件与现有双库，不把论文的参数、等值线数值或误差条套用到这些数据。

## 3. RMS 的明确计算定义

对每个实际数据库格点 `(Te, ne)` 和每个组，计算：

```text
theory_i = 理论谱在实验横轴 E_i 上的线性插值
RMS(Te, ne) = sqrt(sum((theory_i - experiment_i)**2) / N)
```

这里使用普通、未按测量误差加权的 RMS，单位与输入的归一化强度相同。
不除以实验强度，不乘 100，不除以最小 RMS，也不将残差按自由参数数目修正。
输入没有逐点测量误差，因此不计算 chi-square、约化 chi-square 或统计置信区间。

计算流程与约定：

1. 理论的 `E_eV` 与对应谱值、mask 配对排序。实验两列也配对排序。横轴为光子能量位移，**能量损失为负**，不再翻转。
2. 实验中的完全相同能量值默认按该能量的强度均值合并，再让每个保留能量点等权参与 RMS。
   三份原文件各 1205 行、727 个不同能量，默认合并后每组用 727 点。
   这一步用于处理同一描图横坐标下的多个纵坐标，不平滑相邻能量，也不重新峰值归一。
3. `--duplicates keep` 可保留全部 1205 行，此时每一原始行等权，相同能量的多个点会增加该位置权重。
   两种策略的结果可能不同。若这些点代表独立测量而非描图笔画，应按实际采样含义选用；默认策略不是论文已确认的做法。
4. 保留实验负值。不进行背景平移、裁负、拟合幅度、拟合能量零点或再次卷积；直接使用你提供的归一化强度。
5. 将理论线性插值到实验点，不把实验插值到理论网格，不外推。
   插值两端都必须满足 `rms_valid_mask=True`；不能跨越无效 mask 区间。
6. 用 A、B 全部格点的共同有效覆盖约束实验点，同一张图的每个格点使用**完全相同的一组点**。
   三份现有实验横轴相同，因此三组也使用同样的横轴；换成横轴不同的文件时，各组保留自己的实验点，实际点数写入摘要。
7. 默认取实验与理论共同有效的完整能区；当前为 `[-57.5819060889, 39.5768750343] eV`。
   这包括峰和谱翼/基线。若研究目标只针对某段红翼，应先按物理目标选择能区，再通过参数显式限制。
8. 最佳点从实际 Te/ne 网格选取，不在参数平面做连续优化。等值线仅对相邻格点的 RMS 作绘图插值，不对 RMS 面进行平滑。

“去弹性”的两组是比较**已经扣除弹性分量的谱**，默认仍使用整个共同能区；不等于自动删除零能量附近的所有点。
改变拟合窗口、重复点处理或输入归一化会改变目标函数，应将不同设置的输出写入不同目录。

## 4. 安装与运行

建议 Python 3.10 或以上。依赖仅为 NumPy、Matplotlib；不要求 SciPy 或 PDF 库。

在已能运行 Python 的环境中：

```powershell
python -m pip install numpy matplotlib
python 'C:\Users\HP\Documents\Harry\XRTS\rms_sif_contours.py'
```

所有默认相对路径都相对于脚本所在目录解析，所以从其他工作目录启动也可以。
运行时不弹 GUI 窗口，图片自动写入 `C:\Users\HP\Documents\Harry\XRTS\rms_sif_results`。
再次运行到相同输出目录会覆盖同名结果文件，输入文件保持不变。

### 当前这台机器已验证的运行方式

本次检查发现系统 `python` 商店启动别名和项目 `.venv` 无法启动；实际使用 Codex 自带 Python 3.12.14，
配合项目原有 `.validation/deps` 中的 Matplotlib 依赖完成计算。没有为本任务安装新依赖。
在这套目录仍存在时，可直接运行：

```powershell
$py = 'C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -c "import sys,runpy; sys.path.insert(0,r'C:\Users\HP\Documents\Harry\XRTS\.validation\deps'); runpy.run_path(r'C:\Users\HP\Documents\Harry\XRTS\rms_sif_contours.py',run_name='__main__')"
```

该依赖路径仅供当前机器复现。程序本身没有写死 Codex Python 或 `.validation` 路径；换机时使用普通 NumPy/Matplotlib 环境即可。

### 修改拟合能区

例如只使用光子位移 `-50` 到 `-10 eV`（仅为参数用法示例，不是本任务已经确定的物理最佳窗口）：

```powershell
python 'C:\Users\HP\Documents\Harry\XRTS\rms_sif_contours.py' --energy-min -50 --energy-max -10 --output-dir rms_sif_results_window
```

在上面的 Codex Python 方式中，把参数追加到命令最后，同样有效：

```powershell
& $py -c "import sys,runpy; sys.path.insert(0,r'C:\Users\HP\Documents\Harry\XRTS\.validation\deps'); runpy.run_path(r'C:\Users\HP\Documents\Harry\XRTS\rms_sif_contours.py',run_name='__main__')" --energy-min -50 --energy-max -10 --output-dir rms_sif_results_window
```

保留原始重复点、提高图片分辨率、调整等值线数量：

```powershell
python 'C:\Users\HP\Documents\Harry\XRTS\rms_sif_contours.py' --duplicates keep --dpi 300 --levels 24 --output-dir rms_sif_results_keep
```

更换实验文件或数据库：

```powershell
python 'C:\Users\HP\Documents\Harry\XRTS\rms_sif_contours.py' --exp-total 'D:\my_data\total.txt' --exp-inelastic-total 'D:\my_data\inelastic_total.txt' --exp-inelastic-peak 'D:\my_data\inelastic_peak.txt' --db-a 'D:\my_db\db_glz_a_sif' --db-b 'D:\my_db\db_glz_b_sif' --output-dir 'D:\my_results\rms'
```

替换实验文件应为两列数值 `(光子位移/eV, 对应归一化强度)`，默认空白分隔，可有以 `#` 开头的注释。
若新文件采用损失为正的横轴，应事先明确转换成当前约定，不要只反转一列的顺序。
也可以修改代码顶部 `DATABASE_A/B`、三个 `EXPERIMENT_*`、`OUTPUT_DIRECTORY`、能区和 `DUPLICATES` 配置。
命令行参数优先于代码顶部配置；完整参数见 `python rms_sif_contours.py --help`。

## 5. 输出文件

三个前缀分别是 `total`、`inelastic_total`、`inelastic_peak`：

| 文件 | 内容 |
|---|---|
| `*_rms_contour.png` | 三张单独的 Te-ne RMS 等值线图；暗色为较低 RMS，星号为网格最低点 |
| `*_scan.csv` | 每个格点的 Te、ne、RMS、使用点数及 NPZ 文件名 |
| `*_grid.npz` | RMS 二维矩阵、温度/密度轴、实际实验点、最佳理论插值及各点来源行数 |
| `*_best_fit.csv` | 实际 RMS 点的实验、理论插值和残差，残差符号为 theory - experiment |
| `*_best_theory_full.csv` | 最佳格点未经插值的整条理论谱与有效 mask |
| `best_fit_comparison.png` | 三组最佳理论与实验的谱线叠图 |
| `best_fits.csv` | 三组最低点汇总 |
| `summary.json` | 公式、能区、重复点策略、点数、最佳元数据、物理条件、输入和代码哈希、数据库摘要及软件版本 |

`*_grid.npz` 中 `rms.shape == (len(ne_m3), len(Te_eV))`，即**第一维为密度，第二维为温度**。
`experiment_source_rows` 表示一个保留点来自几行实验数据；默认均值模式下它**不参与额外加权**。
CSV 使用带 BOM 的 UTF-8，便于 Windows 软件读取。

等值线每张采用各自的线性色阶，不能根据三张图的同一种颜色比较数值。
更关键的是第三组的强度归一尺度不同，三组 RMS 的绝对大小不宜直接作为“哪一种方法更正确”的排名。
图上的点划线只标记最佳格点的温度与密度，**不是误差条或置信区间**。
纵轴采用 `10^29 m^-3`，与论文的 `10^24 cm^-3` 刻度数值不同。
换算为 `ne[cm^-3] = ne[m^-3] / 10^6`，所以 `10^29 m^-3 = 10^23 cm^-3`。

读取数值示例：

```python
import numpy as np

with np.load('rms_sif_results/total_grid.npz', allow_pickle=False) as d:
    te = d['Te_eV']
    ne = d['ne_m3']
    rms = d['rms']
    i_ne, i_te = np.unravel_index(np.argmin(rms), rms.shape)
    print(te[i_te], ne[i_ne], rms[i_ne, i_te])
```

## 6. 当前默认运行的数值结果

两库分别有 1209 个文件，实际网格均为：

- `Te = 1, 2, ..., 31 eV`，共 31 点；
- `ne = 1e29 + j*5e28 m^-3, j=0..38`，共 39 点。

默认均值合并重复能量、全共同能区、不拟合附加参数，每组用 727 点：

| 组 | 最佳 Te / eV | 最佳 ne / m^-3 | 最小 RMS |
|---|---:|---:|---:|
| 总谱归一的总谱 | 5 | 3.5e29 | 0.1180503531 |
| 总谱归一后去掉弹性峰 | 5 | 3.5e29 | 0.1130667940 |
| 非弹性谱自身峰值归一 | 10 | 3.0e29 | 0.1713872665 |

精确数值以 `summary.json` 和 `best_fits.csv` 为准。这三个最低点均未落在扫描参数边界。
全库校验发现物理/SIF 元数据一致、峰值归一正确、A 分量关系成立；A/B 因分批重建具有不同代码/运行签名，这不作为物理不一致处理。
当前固定条件包括 `E0=2.96 keV`、`theta=40°`、`Zf=2.2`、`NU_MODE=const`、`nu=5e14 s^-1`、`SII_MODE=dh`。
代码不会将 Zf、角度、碰撞率或仪器响应作为待拟合参数。

这里的最低点只说明：在当前文件、能区、重复点策略与固定数据库条件下，该格点使指定 RMS 最小。
最佳谱线叠图仍有明显的峰位、形状和基线残差；实验噪声、扣弹性误差、归一化误差和模型条件都可能影响最低点。
此外，数据库记录的源能量步长约 `0.1000834 eV`；本任务没有重算加密网格或证明卷积/采样数值收敛。
`rms_valid_mask` 仅检查源积分区间能否覆盖实测 SIF 支撑，不能证明模型准确或数值已经收敛。
因此不从这些图直接给出论文式百分比误差或置信区间，也不把最低点当成已确证的实验物理状态。

## 7. 校验与常见错误

已完成实际双库扫描，并检查三张等值线图及最佳谱线对照图。
另用解析合成数据检查了 RMS、线性插值恢复、负值保留、重复点两种策略，以及 mask 孔洞处不跨区插值。
还对实际输出使用独立方式重算 RMS，核对逐格点表、最佳点及输入来源，结果一致。

脚本会对下列问题报错，不静默选择最近点或补造谱线：

- 数据库没有文件、重复参数点、非完整矩形网格，或 A/B 格点不一致；
- 缺少字段/mask，能量轴重复，谱中含非有限值；
- 同库或 A/B 的固定物理/SIF 条件不同；
- A 总谱峰值或 B 展宽谱峰值不是 1，A 总谱与分量和不一致；
- 所选能区内共同有效实验点不足，或实验格式错误；
- 输出路径落入输入数据库目录。

若提示 B 库峰值不是 1，应先按 `database_sif_reading_guide.md` 判断是否为旧版库；不要在此脚本中悄悄乘缩放系数。
缺少 Matplotlib 时请在实际使用的 Python 环境安装依赖，或使用上面已经验证的本机运行方式。
