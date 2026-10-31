# Glenzer 双库后处理

> 本文件下述命令保留原来的 **8.7 eV 高斯** 流程。
> 改用根目录 `sif_0.txt` 时请阅读 `measured_sif_usage.md`：
> `run_glz_sif.py` 重建使用实验 SIF 的双库；`broaden_glz_free_sif.py` 可只处理旧 B 库。
> 新 A 库的弹性项随实测 SIF 形状变化，不应再套用此处的高斯扣峰脚本。

在项目根目录运行（Windows）：

```powershell
& .\RTS_calculating\.venv\Scripts\python.exe .\broaden_glz_free.py
& .\RTS_calculating\.venv\Scripts\python.exe .\subtract_glz_elastic.py
```

## 1. 非弹性库展宽

输入 `database/db_glz_b`，输出 `database/db_glz_b_broadened`。以 FWHM=8.7 eV 的归一高斯响应做式(54)卷积，不调用物理引擎重新造谱。

原 `E_eV`、`S_free`、`meta_json` 保留，新增：

- `S_free_broadened`：展宽后谱。**没有再次峰值归一化**，卷积后峰值通常小于 1。
- `rms_valid_mask`：距源能区两侧至少 4 sigma 的内部点。输入只有有限能区，窗外按零处理，边缘结果有截断影响。内部掩码只是减少边缘偏差，不能恢复未知窗外信号。
- `processing_json`：本次处理参数，与原始元数据区分。

近零频删点造成的缺口用线性插值补齐。输出能量顺序与原库相同。处理清单是 `processing_manifest.csv`。

## 2. 总谱扣除高斯弹性主峰

输入 `database/db_glz_a`，输出 `database/db_glz_a_inelastic`。总谱已经展宽，**不再次卷积**。

在零频附近 ±13.05 eV 拟合高斯峰加线性背景：

```text
model(E) = A*exp(-(E-center)^2/(2*sigma^2)) + b0 + b1*E
```

固定 FWHM=8.7 eV，中心允许 ±2 eV，采用稳健最小二乘；线性项用于吸收局部非弹性尾部，扣除时**只减去高斯项**。这是一种近似识别，不保证精确恢复真实非弹性分量，尤其在低温/低密度等峰重叠条件下。

保留原 `E_eV`、`S_tot`、`meta_json`，新增：

- `S_elastic_fit`：完整能区上的拟合高斯。
- `S_inelastic`：`S_tot - S_elastic_fit`，保持总谱原有归一化尺度。
- `fit_json`：峰高、中心、宽度、拟合误差、负残差及告警。

不重新归一化、不裁负数、不扣线性背景。`fit_manifest.csv` 汇总每条谱的诊断，告警行应先检查再纳入 RMS。文件成功输出不等于分离可靠。

## 试运行与后续 RMS

两脚本支持 `--limit 3`、`--input PATH`、`--output PATH`、`--fwhm 8.7`；扣峰脚本另支持 `--fit-window 13.05`。禁止输出目录与源库相同。可对指定输出目录重复运行重新生成。

```python
import numpy as np
with np.load('database/db_glz_b_broadened/case_T12_N04.npz') as d:
    E = d['E_eV']
    free_broad = d['S_free_broadened']
    valid = d['rms_valid_mask']
with np.load('database/db_glz_a_inelastic/case_T12_N04.npz') as d:
    inelastic_residual = d['S_inelastic']
    elastic_fit = d['S_elastic_fit']
```

**两种输出不共享幅度尺度**：B 库原来按自由电子峰归一，A 库按总谱峰归一。因此不能直接对两库输出做 RMS 解释为物理分量误差。与实验做形状 RMS 时，先按同一能区与同一规则缩放理论/实验；若要保留总谱相对强度，需额外恢复 B 库相对总谱的比例。当前脚本按要求保留处理前的尺度，没有擅自重新归一化。

以上仅处理数据库，不会自动处理扣图实验数据，也不自动进行 RMS。
