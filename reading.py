import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_reading import load_case

plt.figure(figsize=(8, 5))
for te in (10, 30, 50):
    cc = load_case(rho=10, te=te, th=160)
    plt.plot(cc["E"], cc["S_tot"], label=f"Te={te} eV")
plt.xlabel("Photon energy shift (eV)")
plt.ylabel("S(k,ω) (arb. units)")
plt.title("db_v1 demo: rho=10 g/cm³, θ=160°")
plt.legend()
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_v1_读取演示.png")
plt.savefig(out, dpi=110)
print(f"\n演示图已存：{out}")
