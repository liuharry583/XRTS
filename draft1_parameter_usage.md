# draft3 参数接口与派生状态

## 模型选择与代码结构

代码顺序：常数 -> 用户配置/状态接口 -> 物理表达 -> 仪器响应 -> 模型选择 -> 计算/绘图。主要选择集中在文件前部。

| 参数 | 可选值 | 作用 |
| --- | --- | --- |
| `ELECTRON_METHOD` | `integral` / `maxwell` / `auto` | 有限温 Fermi-Dirac 积分 / 非简并 Maxwell 解析 / 历史几何阈值自动策略 |
| `NU_MODE` | `none` / `const` / `bma` | 无碰撞 / 手填常数 / BMA 计算代表碰撞频率 |
| `SII_MODE` | `dh` / `arkhipov` | 弹性项的离子静态结构因子 |
| `INSTRUMENT_FWHM_eV` | 正数 [eV] | 唯一的整谱响应宽度输入，表示半高全宽 |

```python
import draft3 as pc
cfg = pc.PhysicsConfig(
    ELECTRON_METHOD="integral",
    NU_MODE="none",
    SII_MODE="dh",
)
E, omega, free_raw, total_observed, k = pc.compute(
    40.0, -100.0, 100.0, nw=1000, config=cfg
)
```

旧调用中的显式 `method` 仍有效，优先于配置；省略则使用 `ELECTRON_METHOD`。`auto` 按 `k/kappa>2` 选择 Maxwell，**不检查简并条件**。Maxwell 与积分分支使用不同分布假设，不是一般情况下完全等价的两种求值方式。BMA 内部及屏蔽云仍使用既有 Lindhard 积分。

碰撞参数优先级：

- `none`：实际 `nu=0`，忽略手填频率。
- `const`：`nu_rad_s` 非 `None` 时优先，否则取 `NU_FRAC*wp`；允许零值。
- `bma`：实际取 `nu_BMA(k, wp, ...)`，**忽略 `NU_FRAC` 和 `nu_rad_s`**；填零不会关闭 BMA。本实现仅在 `omega=wp` 评估一次，整谱共用，不是逐谱点动态 BMA。内部 DH 屏蔽不受 `SII_MODE` 影响。

查看真正采用的数值：

```python
state = pc.build_state(cfg)
k = pc.k_of_theta(40.0, config=cfg)
nu_used = pc.collision_frequency(k, state)  # rad/s
```

其它设置相同，`none` 与 `const, NU_FRAC=0, nu_rad_s=None` 完全一致。把 `const` 的 `nu_rad_s` 设成计算出的 BMA `nu_used`，也会复现本实现的 BMA 谱。BMA 函数的评估频率 `omega` 与碰撞频率 `nu` 不同，`omega=0` 不表示关闭碰撞。

保留 `USE_CHIHARA_ION=False` 及旧弹性函数仅用于兼容；推荐使用默认 Chihara 离子项。

## 2026-10-26：整谱仪器响应（Glenzer & Redmer 2009，式 53–54）

`compute()` 的总谱现在同时展宽弹性项和自由电子项：

```text
S_tot = A(k) * g(omega) + Zf * integral[S_free(omega') * g(omega-omega') d omega']
```

离子本征项采用 A(k) delta(omega)，其卷积解析计算为 A(k) g(omega)，不会重复展宽。
自由电子卷积使用扩展至输出能区外 ±8 sigma 的源网格，并在源网格中补上零频极小邻域的可去奇点；不对有限输出窗口重新归一化。

推荐设置 `INSTRUMENT_FWHM_eV`，它是**探针带宽与探测系统分辨率合成后的高斯 FWHM**，单位 eV；两者都近似高斯时，`FWHM_tot = sqrt(FWHM_probe**2 + FWHM_detector**2)`。

```python
cfg = pc.PhysicsConfig(INSTRUMENT_FWHM_eV=18.0)
E, omega, free_raw, total_observed, k = pc.compute(
    40.0, -100.0, 100.0, nw=2000, config=cfg
)
# 或兼容全局接口：pc.configure(INSTRUMENT_FWHM_eV=18.0)
```

上述 18 eV 仅为接口示例，不是已标定的实验分辨率。第三个返回值仍为**未展宽、单电子归一**的自由电子谱，第四个是**统一展宽后的总谱**；频移坐标无需再加探针中心频率。

现在只接受 `INSTRUMENT_FWHM_eV`，必须为有限正数，不能填 `None`。默认值为 42.386760810557085 eV，等价于原标准差 18 eV，保持原默认展宽效果；这仍是占位设置，不是实验标定值。内部自动换算 `sigma_E = FWHM / sqrt(8*ln(2))`。

旧参数 `ELASTIC_WIDTH_eV` 已删除。迁移旧配置时，将原标准差乘以 `sqrt(8*ln(2))` 后传给 `INSTRUMENT_FWHM_eV`，不要直接照抄旧数值。

这是有意改变总谱的物理处理；下文参数重构期间的旧谱逐位相等验证和旧数据库复算，不能作为此版本总谱不变的保证。自由电子本征公式、离子权重、碰撞模型未改变。

本次仅重构参数管理，不更改 Lindhard/RPA/Mermin、BMA、DH、Arkhipov 等物理公式。修改前完整源码保存在 `.validation/parameter_state/draft3_before.py`。

## 推荐：每条谱显式传配置

```python
from dataclasses import replace
import draft3 as pc

base = pc.PhysicsConfig(
    E0_keV=2.96, ZA=4.0, Zf=2.0,
    Te_eV=10.0, ne=6.0e29,
    NU_MODE='const', NU_FRAC=0.0,
    SII_MODE='arkhipov', INSTRUMENT_FWHM_eV=4.709640090061899,
)
for te in (5.0, 10.0, 15.0):
    config = replace(base, Te_eV=te)
    state = pc.build_state(config)
    E, omega, free, total, k = pc.compute(
        40.0, -100.0, 100.0, nw=2000, method='auto', config=config
    )
    # 保存时从同一个 config/state 生成 meta_json 和派生量，避免标签与计算不符。
```

上例的谱网格和物理模型选项只是接口示例，不是已收敛的最佳拟合方案。配置不可变，可用 dataclasses.replace 生成下一条谱。显式配置不会改动模块基础参数；不同算例互不污染。

## 兼容：集中设置或继续给模块基础参数赋值

```python
pc.configure(E0_keV=2.96, Te_eV=10.0, ne=6e29, ZA=4.0, Zf=2.0)
# 此时 lam0/T/wp/kap/nu/mu 均已更新。
E, omega, free, total, k = pc.compute(40.0, -100, 100)

pc.E0_keV = 5.0
# 基础参数直接赋值兼容；下次 compute/k_of_theta 自动同步。
k = pc.k_of_theta(40.0)
```

需要立即读取兼容全局派生值时，使用 configure 或 refresh_state，不能在直接赋值后、调用同步接口前读取旧快照。

T、lam0、wp、kap、nu、mu 不再作为 compute 的独立输入：每次从基础参数重建，并按完整配置缓存。不要继续手工计算它们，也不要以赋值 pc.nu 控制碰撞频率。比例碰撞频率用 NU_FRAC；确需固定绝对频率时，传 `PhysicsConfig(..., nu_rad_s=1e14)`，单位 rad/s。NU_MODE='bma' 时仍按原公式计算局部 nu。

ne 单位为电子密度 m^-3，不能直接传质量密度。真空/Te=0/角度=0 的零填充或奇点仍应由跑批层处理，物理计算入口拒绝这些输入。E 数组保持原计算约定：omega 升序、E 降序；画图/拟合若要求 E 升序，需对 E 与谱值成对排序。

## 历史验证（不适用于当前接口）

```powershell
& .\RTS_calculating\.venv\Scripts\python.exe .\.validation\parameter_state\validate_parameter_state.py
```

上面的脚本属于早期参数重构验证，依赖旧参数及旧总谱基线，不能直接作为当前版本验收。当前接口仅接受 FWHM，旧数据库复算也需先转换宽度参数。
