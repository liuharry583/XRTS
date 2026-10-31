# 实验 SIF 双库：文件、字段与读取说明

检查日期：2026-10-29。项目目录：`C:\Users\HP\Documents\Harry\XRTS`。
本次检查两个新库均有 **1209 个 `case_*.npz` 文件**。
本说明和 `reading.py` 只读取已有结果，不启动跑批或修改数据库。

## 1. 两个库分别是什么

| 数据库目录 | 用途 | 与实验比较优先读取的字段 |
|---|---|---|
| `database/db_glz_a_sif` | 总谱及其弹性、非弹性分量，三者共用总谱的强度尺度 | 整条谱用 `S_tot`；扣弹性后用 `S_inelastic` |
| `database/db_glz_b_sif` | 自由电子原始谱和展宽谱，便于单独研究非弹性峰 | `S_free_broadened`，而不是未展宽的 `S_free` |

两库均由实验 `sif_0.txt` 的响应展宽，不再使用 8.7 eV 高斯宽度。
用户已确认输入 SIF 是“能量损失为正”；生成时已转换为 draft 的光子位移
坐标。**数据库里的 E_eV 仍然是能量损失为负**，不要再把 SIF 对这些谱卷一次。

### 算例文件名

每个温度、密度组合对应一个 NumPy 压缩存档 `.npz`，不是单条纯文本曲线。
两库文件一一对应，例如 `case_T12_N04.npz`：

- `T12` 表示电子温度 Te=12 eV。
- `N04` 表示密度索引 j=4，**不是密度值4**。
- 电子密度 `ne = 1e29 + j*5e28 m^-3`，因此 N04 对应 `3e29 m^-3`。
- Te=1..31、j=0..38，所以完整网格为 31×39=1209 组。
- 文件名不包含散射角；本双库为固定 40°、探针能量 2.96 keV。

## 2. A 库每个 npz 的内容

| 字段 | 类型/含义 |
|---|---|
| `E_eV` | 一维能量网格，单位 eV；文件内通常降序，红移/损失为负 |
| `S_tot` | 已用实测 SIF 展宽的总谱，当前输出能区内最高值为1 |
| `S_elastic` | 已展宽的模型弹性/离子项；响应形状来自 SIF，不是重新拟合的高斯 |
| `S_inelastic` | 已展宽的非弹性项，**已包含 Zf 权重** |
| `rms_valid_mask` | 与能量同长度的布尔数组，说明卷积是否覆盖完整实测核支撑区间 |
| `meta_json` | JSON 字符串，含物理参数、SIF来源/哈希/坐标处理、源积分步长及归一化尺度 |

三项使用**相同的总谱归一化因子**，因此：

```text
S_tot = S_elastic + S_inelastic
max(S_tot) = 1
```

`S_elastic` 与 `S_inelastic` 各自的最大值不一定为1。读取时不要分别再归一化，
否则会丢掉弹性和非弹性的相对强度。

**A 库已经直接保存扣除模型弹性后的非弹性项。**不用对它再运行旧的高斯扣峰
脚本；这里也没有从实验数据里替你扣峰，实验侧的分离仍需独立处理。

## 3. B 库每个 npz 的内容

| 字段 | 类型/含义 |
|---|---|
| `E_eV` | 与同编号A库相同的能量网格，损失为负 |
| `S_free` | **未展宽**的自由电子谱，自身最高值归一为1 |
| `S_free_broadened` | 先用实验 SIF 展宽，再按展宽后自身峰值归一化为1 |
| `rms_valid_mask` | 卷积源区间覆盖掩码 |
| `meta_json` | 与同编号A库对应的物理及处理元数据 |

两条自由电子谱没有额外乘 Zf，分别按自身最大值归一化。
B库用于单独比较非弹性形状时，仍读 `S_free_broadened`，其峰值已经为1，
读取器无需修改。此规则自2026-10-30代码更新生效；旧库必须重算后才采用新规则。

**A.S_inelastic 与 B.S_free_broadened 并非相同幅度尺度。**设元数据中的
`Cb = free_broadened_peak_before_normalization`、`Ct = total_peak_before_normalization`，
同一算例、同一引擎参数下的关系为：

```text
A.S_inelastic = Zf * Cb / Ct * B.S_free_broadened
```

因此不能把B的谱不经尺度处理就直接减到A总谱上。做“整条RMS”和“非弹性RMS”
且希望沿用总谱强度尺度时，直接使用A库的 `S_tot` 和 `S_inelastic` 最清楚。

## 4. 数据库附属文件

| 文件 | 作用 |
|---|---|
| 各库 `run_config.json` | 本批次配置、代码哈希、SIF哈希、归一化规则及运行签名 |
| 各库 `sif_kernel_used.txt` | 实际使用的单位面积核，两列为光子位移/eV和响应/每eV；这里已经转换坐标 |
| `database/db_glz_sif_manifest.csv` | 两库共享的逐算例运行日志，含成功、失败或跳过状态以及耗时；断点续跑会追加日志 |

共享manifest可能含同一算例的多次运行记录，所以**日志行数不等于独立格点数**。
新 `reading.py` 根据实际存在的npz及其元数据索引GLZ库，避免把重复日志当成
重复算例。原有库仍按它们自己的 `manifest.csv` 读取。

## 5. reading.py 怎样用

### 方法一：改文件顶部配置后直接运行

默认配置为A库、Te=12 eV、ne=3e29 m^-3，默认画总谱：

```python
DATABASE = "db_glz_a_sif"
TEMPERATURES = (12,)
GLZ_NE_M3 = 3e29
SPECTRUM = None
LOSS_AXIS = False
```

改成 `DATABASE = "db_glz_b_sif"` 且保持 `SPECTRUM = None`，会自动画
`S_free_broadened`。A库需要一张图看三项时，设置：

```python
SPECTRUM = ("S_tot", "S_elastic", "S_inelastic")
```

命令行参数优先于文件顶部配置。`TEMPERATURES=None` 对GLZ默认选择Te=12；
对原质量密度库默认(10,30,50)，对原细网格库默认(5,10,20)。

### 方法二：命令行选库、选谱

先在PowerShell定义两个绝对路径（后面的命令共用）：

```powershell
$py = 'C:\Users\HP\Documents\Harry\XRTS\RTS_calculating\.venv\Scripts\python.exe'
$reader = 'C:\Users\HP\Documents\Harry\XRTS\reading.py'
```

查看已有温度、密度、角度和字段：

```powershell
& $py $reader --db db_glz_a_sif --list
& $py $reader --db db_glz_b_sif --list
```

查看单个算例的字段、峰值、SIF信息和完整元数据，不弹图：

```powershell
& $py $reader --db db_glz_b_sif --te 12 --ne 3e29 --info
```

A库画总谱和分量，B库画已展宽/未展宽的非弹性谱：

```powershell
& $py $reader --db db_glz_a_sif --te 12 --ne 3e29 --spectrum S_tot S_elastic S_inelastic
& $py $reader --db db_glz_b_sif --te 12 --ne 3e29 --spectrum S_free_broadened S_free
```

多温度对比与论文横轴方向：

```powershell
& $py $reader --db db_glz_a_sif --te 5 12 20 --ne 3e29 --spectrum S_inelastic --loss-axis
```

`--loss-axis` 只把显示/导出的横轴变成能量损失为正，并重新排序；不改变
物理谱值，不修改npz。默认横轴为库内光子位移。所有曲线的横轴和谱值都会配对
排序，不会单独翻转一列。

### 导出CSV与图片

```powershell
& $py $reader --db db_glz_a_sif --te 12 --ne 3e29 --spectrum S_tot S_elastic S_inelastic --loss-axis --export 'C:\Users\HP\Documents\Harry\XRTS\output\A_T12_N04.csv' --save 'C:\Users\HP\Documents\Harry\XRTS\output\A_T12_N04.png' --no-show
```

CSV包含温度、密度、角度、能量、所选各谱列及 `rms_valid_mask`。
兼容读取没有该掩码的旧库时，CSV此列留空，不推断旧库的卷积有效性。
多温度时各温度按行依次写入，使用Te列区分，不强行插值到同一网格。
`--no-show` 不弹窗口；仅查看文本可用 `--info`，只导出数值可加 `--info --export PATH`。
`--valid-only` 仅保留mask为True的点。读取器**不重新归一化、不裁负值、
不重采样、不做RMS、不再次展宽**。

### 方法三：在自己的Python程序里读取

在项目目录运行，或先将项目目录加入 `sys.path`：

```python
import sys
sys.path.insert(0, r'C:\Users\HP\Documents\Harry\XRTS')
from reading import load_case

a = load_case('db_glz_a_sif', 12, ne=3e29)
b = load_case('db_glz_b_sif', 12, ne=3e29)
E = a['E_eV']               # load_case返回升序光子位移
total = a['S_tot']
elastic = a['S_elastic']
inelastic = a['S_inelastic'] # 已展宽，含Zf，与total同尺度
free_raw = b['S_free']
free_broad = b['S_free_broadened']
meta = a['meta']             # 已解析好的Python字典
valid = a['rms_valid_mask']
```

要看原文件字段，直接 `np.load(..., allow_pickle=False)` 后查看 `.files`。
此时 `meta_json` 仍是字符串，需要 `json.loads(str(d['meta_json']))`，原始数组
顺序也保持文件内降序。不要把npz当成txt用 `np.loadtxt`。

## 6. 比较实验前的检查

- **整条RMS**：实验总谱与A库 `S_tot` 对比。
- **非弹性RMS**：扣除实验弹性响应后的实验谱，与A库 `S_inelastic` 或B库
  `S_free_broadened` 对比。选择哪一个要配套明确幅度尺度；不能混用归一化规则。
- 统一正负方向和能量零点，再在同一能区将理论插值到实验采样点；读取器不替代拟合流程。
- 新谱已经包含实测SIF，不能重复卷积。实测核测量范围外按零处理，文件中的负值
  默认置零并记录；核翼部、基线误差和实验扣峰误差仍然可能影响RMS。
- `rms_valid_mask=True` 只表示源积分覆盖完整实测核支撑，不是物理模型准确、
  实验无噪声或源网格已完全收敛的保证。
- 可用Te/ne必须是实际格点。选不到时查看 `--list`；读取器会报错，不偷换成附近格点。

生成参数和重跑方法另见项目文件 `measured_sif_usage.md`。
只重建B库采用 `run_glz_sif.py --only-b --rebuild-b`；不会写A库。
