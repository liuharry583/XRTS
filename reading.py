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
plt.show()
