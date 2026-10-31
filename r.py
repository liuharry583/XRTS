import sys
import numpy as np

sys.path.insert(0, r"C:\Users\HP\Documents\Harry\XRTS")
from reading import load_case
from matplotlib import pyplot as plt

a = load_case("db_glz_a_sif", 12, ne=3e29)
b = load_case("db_glz_b_sif", 12, ne=3e29)
E = a["E_eV"]  # load_case返回升序光子位移
free_a = a["S_inelastic"]
free_b = b["S_free_broadened"]

data = np.loadtxt("exp_noelastic_2_negative.txt")[:, :]

plt.figure(figsize=(8, 6))

plt.plot(E, free_b, label="inelastic")
plt.plot(data[:, 0], data[:, 1], label="experiment")
# plt.plot(E, free_b, label="free")

plt.xlabel("E (eV)")
plt.ylabel("S")
plt.legend()
plt.show()
