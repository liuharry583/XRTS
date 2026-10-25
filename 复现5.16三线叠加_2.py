# reproduce_fig5_15_fixed.py
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# ---------- 修复 matplotlib 中文字体显示 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# ---------- 加载原代码模块 ----------
spec = importlib.util.spec_from_file_location("chihara_mod", "ρ与Sii添加验证.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ---------- 设置物理参数 ----------
Te_eV = 20.0            # 对应您截图中的 Te=20 eV
ne = 2.5e23 * 1e6       # 对应您截图中的 ne=2.5e23 cm^-3
Zf = 2.5                # 对应您截图中的 Zf=2.5
ZA = 6.0
E0_keV = 4.75
NU_FRAC = 0.0
ELASTIC_WIDTH_eV = 7.2  # 对应您截图中的 sigma=7.2 eV
USE_CHIHARA_ION = True

mod.Te_eV = Te_eV
mod.ne = ne
mod.Zf = Zf
mod.ZA = ZA
mod.E0_keV = E0_keV
mod.NU_FRAC = NU_FRAC
mod.ELASTIC_WIDTH_eV = ELASTIC_WIDTH_eV
mod.USE_CHIHARA_ION = USE_CHIHARA_ION

# 重新计算派生量
mod.T = mod.Te_eV * mod.EV / mod.kB
mod.lam0 = mod.h * mod.c / (mod.E0_keV * 1e3 * mod.EV)
mod.wp = np.sqrt(mod.ne * mod.e**2 / (mod.eps0 * mod.me))
mod.kap = np.sqrt(mod.ne * mod.e**2 / (mod.eps0 * mod.kB * mod.T))
mod.nu = mod.NU_FRAC * mod.wp
mod.mu = mod.chem_pot(mod.T, mod.ne)

# ---------- 计算散射核 S_tot(shift) ----------
theta_deg = 109.0
E_shift_lo, E_shift_hi = -250.0, 100.0
nw = 2000
E_shift, w, S_free_arr, S_tot_shift, k = mod.compute(
    theta_deg, E_shift_lo, E_shift_hi, nw=nw, method="auto"
)

# ⚠️【关键修复点】原代码内部对 w 进行了升序排序，导致 E_shift 变成了降序。
# np.interp 要求 x 坐标必须是升序，因此这里需要将其翻转为升序！
E_shift = E_shift[::-1]
S_tot_shift = S_tot_shift[::-1]
S_tot_shift = np.nan_to_num(S_tot_shift) # 防止极值点产生 NaN

# ---------- 三条背光谱线 ----------
lines = [
    {"E_in": 4706.0, "I": 0.3, "color": "C0", "label": "类Li线 ~4706 eV"},
    {"E_in": 4728.0, "I": 0.7, "color": "C1", "label": "互组合线 ~4728 eV"},
    {"E_in": 4750.0, "I": 1.0, "color": "C2", "label": "共振线 ~4750 eV"},
]

E_out = np.linspace(4600.0, 4840.0, 2400)

# 每条线的散射谱：shift = E_out - E_in
S_lines = []
for ln in lines:
    shift = E_out - ln["E_in"]
    # 现在 E_shift 是升序的，插值不会出错
    S_line = ln["I"] * np.interp(shift, E_shift, S_tot_shift, left=0.0, right=0.0)
    S_lines.append(S_line)

S_sum = np.sum(S_lines, axis=0)

# 源谱（未散射）高斯包络
sigma = ELASTIC_WIDTH_eV
S_source = np.zeros_like(E_out)
for ln in lines:
    S_source += ln["I"] * np.exp(-0.5 * ((E_out - ln["E_in"]) / sigma) ** 2)

# 归一化
S_sum_norm = S_sum / np.max(S_sum)
S_source_norm = S_source / np.max(S_source)

# ---------- 绘图 ----------
plt.figure(figsize=(10, 6))

# 1. 画三条散射分量
for ln, S_line in zip(lines, S_lines):
    plt.plot(E_out, S_line / np.max(S_sum), color=ln["color"], lw=1.2, alpha=0.9,
             label=f"散射分量: {ln['label']}")

# 2. 画总散射谱（黑线）
plt.plot(E_out, S_sum_norm, 'k-', lw=2.0, label="总散射谱（三线叠加）")

# 3. 画源谱参考（灰色虚线）
plt.plot(E_out, S_source_norm, '--', color='gray', lw=1.5, label="源谱（未散射，归一化）")

plt.xlabel("光子能量 (eV)", fontsize=12)
plt.ylabel("归一化强度 (arb. units)", fontsize=12)
plt.title(f"复现论文图5.15/5.16：Ti He-α三线独立散射与叠加\n"
         f"Te={Te_eV:g} eV, ne={ne/1e6:.1e} cm$^{{-3}}$, Zf={Zf:g}, "
         f"θ={theta_deg:g}°, α={mod.kap/k:.2f}, σ_elastic={ELASTIC_WIDTH_eV:g} eV",
         fontsize=13)
plt.xlim(4600, 4840)
plt.ylim(-0.02, 1.1)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.grid(alpha=0.3)
plt.show()