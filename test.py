from dataclasses import replace
from matplotlib import pyplot as plt
import numpy as np
import draft as pc

cfg = pc.PhysicsConfig(
    Te_eV=12.0,
    ne=3.0e29,
    Zf=2.2,
    ZA=4.0,
    E0_keV=2.96,
    ELECTRON_METHOD="integral",
    NU_MODE="const",
    nu_rad_s=5.0e14,
    SII_MODE="dh",
    INSTRUMENT_FWHM_eV=8.7,
)
E, w, S_free, S_tot, k = pc.compute(
    40.0, -70.0, 45.0, nw=1800, config=cfg, normalize=True
)
# 仿原文的 finite-k 处理：对 θ = 25…55 取 5 个角度各算一次再平均（我试跑就是这么做的）
# 画 Glenzer 式横轴：光子能量 keV = 2.96 - (损失 eV)/1000，其中损失 = -E

data = np.loadtxt("./RTS_calculating/exp_data.txt")

plt.figure(figsize=(8, 6))
plt.plot(E, S_tot)

plt.xlabel("Energy [eV]")
plt.ylabel("Dynamic structure factor S(k, ω)")
plt.title(f"ZA={cfg.ZA}, Zf={cfg.Zf}, ne={cfg.ne:.2e} m^-3")
plt.show()
