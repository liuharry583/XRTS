from draft2 import plot_spectrum, compute
from matplotlib import pyplot as plt

if __name__ == "__main__":
    theta_deg = 80
    E_lo, E_hi = -1, 1
    E, w, S, S_tot, k = compute(theta_deg, E_lo, E_hi, method="integral")
    plt.figure()
    plt.plot(E, S_tot, "k-", lw=1.2)
    plt.axvline(0, color="gray", ls="--", lw=0.8)
    plt.xlabel("Photon energy shift (eV)")
    plt.ylabel("S(k,ω) (arb. units)")
    plt.title(f"θ={theta_deg}°")
    plt.xlim(E_lo, E_hi)
    plt.ylim(0, None)
    plt.show()
