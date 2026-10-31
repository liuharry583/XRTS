# 用 sif_0.txt 替换高斯响应

生成后如何查看字段、画谱、导出CSV：见 `database_sif_reading_guide.md`。
`reading.py` 已支持新双库；A默认读总谱，B默认读已展宽的自由电子谱。

新增代码只读原物理引擎 `draft.py`；原引擎、原双库和高斯后处理脚本保持原样。
默认读取项目根目录的 `sif_0.txt`，而不是归档目录或旧的 `sif_data.txt`。

## 已确认的坐标方向

所有输出 `E_eV` 延续 draft 的光子能量位移：红移/能量损失在负侧。
用户已确认 `sif_0.txt` 第一列采用论文约定：**能量损失为正**。
因此默认 `--sif-axis energy-loss`，代码会将响应横轴转换为光子位移
（乘 -1，并连同强度排序）。无需手动修改 sif_0.txt，也不要再手动镜像一次。
以后若换成已镜像、能量损失为负的文件，则显式加 `--sif-axis photon-shift`。

## 响应如何处理

- 读取两列：能量/eV、响应强度。排序，同坐标多次采样取平均。
- 当前文件有少量负强度。默认 `--negative clip` 将负强度置零，并在元数据
  记录原始负值数量、最小值及处理方式。`--negative error` 则遇负值直接报错。
- 不自动减基线、不平滑、不拟合高斯、不把峰重新平移到零。
- 核按面积归一化为 1；离散卷积再修正采样积分误差。不是按峰高归一化。
- 使用实际能量差 R(E-E')，零延迟对齐，不把非对称文件的数组中点当零点。
- 测量范围外响应设为零；实测区间内的正尾部保留。当前文件两端仍有基线
  痕迹，这里不擅自推断基线，真实响应翼部/基线的不确定性仍会影响 RMS。
- 8.7 eV 不再作为展宽参数；宽度和非对称形状均来自文件。

## 方案一：重新生成两个数据库（推荐完整比较时使用）

在项目目录的 PowerShell 中运行，先小批检查：

```powershell
& .\RTS_calculating\.venv\Scripts\python.exe .\run_glz_sif.py --pilot --limit 1
```

检查后全量跑：

```powershell
& .\RTS_calculating\.venv\Scripts\python.exe .\run_glz_sif.py
```

输出为 `database/db_glz_a_sif` 和 `database/db_glz_b_sif`，默认共 1209 组。
支持断点续跑；只有核文件哈希、轴方向、负值处理、代码和网格配置一致才会跳过
已完成文件。核或步长改变后，使用新的 `--output-root`；不会混进旧结果。
`--limit` 限制选中列表的前 N 组，不是每次补跑 N 个新 case。
`--pilot` 是 Te=12 整行加四角；也可用 `--te 12 --j 4` 指定格点。

物理参数沿用原 Glenzer 双库：E0=2.96 keV、40 度、Be、Zf=2.2、ZA=4、
integral、const nu=5e14 rad/s、DH、Te=1..31、ne=1e29+j*5e28 (j=0..38)。
本版沿用原引擎零频附近的处理方式；不声称重新验证了这些物理近似。

新 A 库：

- `S_tot`：实验 SIF 展宽后的总谱，全谱最大值为 1。
- `S_elastic`、`S_inelastic`：与 S_tot **共用同一归一尺度**，两项相加等于 S_tot。
  弹性项为模型离子权重乘 SIF；不是再拟合一个高斯峰。
- `S_inelastic` 已包含 Zf 权重，可以直接作为理论的“扣弹性后”谱；无需对新
  A 库再运行原来的高斯扣峰脚本。实验侧如何扣峰仍是独立问题。

新 B 库：

- `S_free`：未展宽、自身峰值为 1，保持旧 B 库字段含义。
- `S_free_broadened`：先计算实测 SIF 展宽，再除以展宽后自身最大值，**峰值为 1**。
  与实验非弹性谱比较仍读取此字段；`S_free` 仍是未展宽且峰值为 1 的谱。

### 只重算 B 库（2026-10-30 更新）

```powershell
& .\RTS_calculating\.venv\Scripts\python.exe .\run_glz_sif.py --only-b --rebuild-b
```

此命令在原 `database/db_glz_b_sif` 路径重算并覆盖全部 B 算例，A 库不读取也不写入。
目录名、文件名、谱字段和 `reading.py` 的用法不变。`--rebuild-b` 必须与
`--only-b` 一起使用，且不能搭配格点/数量筛选，以避免只替换部分归一化口径。
运行中断时再次运行同一命令会从头重算覆盖；完成前不要混用新旧 B 算例。
常规同配置续跑仍可用 `--only-b`，但不同配置的旧库必须明确使用重建选项。
不覆盖旧库的试跑可用 `--only-b --te 12 --j 4 --output-root 新目录`。
当前引擎导入项目根目录 `physical_core.py`（原 draft.py），不修改其物理公式。

两库还保存 `meta_json` 和 `rms_valid_mask`。重算版扩展理论源区间覆盖整个
实测 SIF 支撑区间，因而输出区间的卷积不截断理论源；该掩码预计全为 True。
文件 `sif_kernel_used.txt` 是单位面积核，`run_config.json` 记录来源和哈希，
`database/db_glz_sif_manifest.csv` 记录每次运行的逐 case 状态。

## 方案二：仅给现有 B 库换核（不用重算物理）

```powershell
& .\RTS_calculating\.venv\Scripts\python.exe .\broaden_glz_free_sif.py --limit 3
# 试跑输出需换一个目录，避免与下一次全量输出重名：
& .\RTS_calculating\.venv\Scripts\python.exe .\broaden_glz_free_sif.py --output .\database\db_glz_b_sif_postprocessed_full
```

默认输入 `database/db_glz_b`，输出 `database/db_glz_b_sif_postprocessed`。
保留原字段，新增 `S_free_broadened`、`rms_valid_mask` 和 `processing_json`。
展宽后按自身峰值归一化为1；不会覆盖源库或对已有展宽谱再次卷积。

旧 B 库只有 -70..50 eV 的理论源区间，外侧当零；SIF 翼部较宽，严格全支撑
覆盖的 `rms_valid_mask` 会比原高斯版窄。它不能保证恢复窗外未知理论信号。
如果你的 RMS 能区超出 True 区间，应使用方案一扩源重算，而不是忽略掩码。
旧 A 库已经做过高斯展宽，不能把它再卷一次 SIF 当成换核，应使用方案一。

## 单独调用引擎

```python
import draft_sif
from sif_response import MeasuredSIF

sif = MeasuredSIF.load('sif_0.txt', axis='energy-loss', negative='clip')
cfg = draft_sif.PhysicsConfig(Te_eV=12, ne=3e29, E0_keV=2.96,
    ZA=4, Zf=2.2, ELECTRON_METHOD='integral', NU_MODE='const', nu_rad_s=5e14)
E, omega, free_raw, total, k = draft_sif.compute(40, -70, 50, nw=1200,
    config=cfg, sif=sif)
# 五项返回值兼容 draft；free_raw 仍为未展宽。完整分量调用：
components = draft_sif.compute_components(40, -70, 50, nw=1200, config=cfg, sif=sif)
```

`PhysicsConfig` 复用原类，因此仍有 `INSTRUMENT_FWHM_eV` 字段；SIF 版完全不使用
这个字段。新数据库元数据不把它当成有效参数。不要将新库再次做仪器卷积。

## 数值检查

更细源网格可用 `--step 0.05 --output-root PATH` 单独试跑，比较峰面积/峰形和
最终 RMS 的收敛。文件哈希相同、曲线平滑不等于数值和物理误差已经消失。

As of 2026-10-29 — measured_sif_usage.md
