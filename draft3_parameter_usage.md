# draft3 参数接口与派生状态

本次仅重构参数管理，不更改 Lindhard/RPA/Mermin、BMA、DH、Arkhipov 等物理公式。修改前完整源码保存在 `.validation/parameter_state/draft3_before.py`。

## 推荐：每条谱显式传配置

```python
from dataclasses import replace
import draft3 as pc

base = pc.PhysicsConfig(
    E0_keV=2.96, ZA=4.0, Zf=2.0,
    Te_eV=10.0, ne=6.0e29,
    NU_MODE='const', NU_FRAC=0.0,
    SII_MODE='arkhipov', ELASTIC_WIDTH_eV=2.0,
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

## 验证

```powershell
& .\RTS_calculating\.venv\Scripts\python.exe .\.validation\parameter_state\validate_parameter_state.py
```

检查默认谱回归、手工正确同步旧版本与新配置的等价性、基础参数改变后的自动同步、独立配置隔离、固定碰撞频率、错误输入、两种离子模型/BMA 路径，以及保存的 db_v2 样例复算。原同学 physics_core.py 与既有数据库未修改。
