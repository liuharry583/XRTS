# draft 用法：谱生成与峰高归一化

```python
import draft as pc

cfg = pc.PhysicsConfig(
    ELECTRON_METHOD="integral",  # 或 maxwell / auto
    NU_MODE="none",             # 或 const / bma
    SII_MODE="dh",              # 或 arkhipov
    INSTRUMENT_FWHM_eV=12.0,     # 示例：整谱有效响应 FWHM [eV]
)
E, omega, S_free, S_tot, k = pc.compute(
    40.0, -100.0, 100.0, nw=1000, config=cfg
)
```

默认开启峰高归一化，返回格式不变：

- `S_free`：保持原来的未展宽自由电子谱定义，除以当前输出能区内的最大值，故最大值为 1。双峰时以其中较高的峰为基准，不分别缩放两个峰。输出能区应覆盖目标峰。
- `S_tot`：先计算整谱展宽，再除以当前输出能区内总谱的最大值，因此总谱最高值为 1，弹性与非弹性强度比保持不变。

两种输出都以各自数组的最高值为 1，不指定必须由弹性峰还是非弹性峰提供最大值。归一化基准取决于输出能区，生成时应覆盖需要比较的峰。

两条返回谱分别采用不同分母，不能用 `S_tot - Zf*S_free` 提取弹性分量。需要物理单位、求和规则检查或分量运算时，关闭归一化：

```python
E, omega, free_raw, total_raw, k = pc.compute(
    40.0, -100.0, 100.0, nw=1000, config=cfg, normalize=False
)
```

绘图时直接选择输出：

```python
pc.plot_spectrum(40.0, -100.0, 100.0, config=cfg, spectrum="S_free")
pc.plot_spectrum(40.0, -100.0, 100.0, config=cfg, spectrum="S_tot")
```

`normalize=False` 同样适用于 `plot_spectrum()`。归一化参考峰必须是有限正数；无法定义时明确报错，不用零值或任意常数替代。
