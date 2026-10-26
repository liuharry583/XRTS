from dataclasses import replace
from matplotlib import pyplot as plt
import draft as pc

base = pc.PhysicsConfig(
    E0_keV=2.96,
    ZA=4.0,
    Zf=2.0,
    Te_eV=10.0,
    ne=6.0e29,
    NU_MODE="const",
    NU_FRAC=0.0,
    SII_MODE="arkhipov",
    INSTRUMENT_FWHM_eV=12.0,  # 等效原标准差 5 eV
)

plt.figure()
E, omega, free, total, k = pc.compute(
    40.0, -100.0, 100.0, nw=2000, method="auto", config=base
)
plt.plot(E, total)

plt.xlabel("Energy [eV]")
plt.ylabel("Dynamic structure factor S(k, ω)")
plt.title(f"ZA={base.ZA}, Zf={base.Zf}, ne={base.ne:.2e} m^-3")
plt.show()
