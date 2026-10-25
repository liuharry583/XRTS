import numpy as np
import matplotlib.pyplot as plt
import RTS计算.physics_core as pc
from scipy.signal import convolve
from scipy.interpolate import interp1d
import os

# ============================================================
# 1. 参数设置
# ============================================================
pc.ZA = 4.0  # 铍
pc.Zf = 2.0
pc.E0_keV = 2.96  # 探针光子能量
THETA_DEG = 40.0  # 散射角
pc.NU_MODE = "ZERO"  # 非集体散射，关闭碰撞，使用 Maxwell 解析解提速

# 拟合能区 (Compton 红翼)
E_LO = -35.0
E_HI = -10.0


# ============================================================
# 2. 数据清理函数
# ============================================================
def clean_data(data):
    x = data[:, 0]
    y = data[:, 1]
    y[y < 0] = 0  # 截断负噪声
    # 去除重复的 X 坐标并取平均
    unique_x = np.unique(x)
    unique_y = np.zeros_like(unique_x)
    for i, val in enumerate(unique_x):
        unique_y[i] = np.mean(y[x == val])
    return unique_x, unique_y


if not os.path.exists("exp_data.txt") or not os.path.exists("sif_data.txt"):
    raise FileNotFoundError("请确保当前目录下有 exp_data.txt 和 sif_data.txt！")

E_exp_raw, S_exp_raw = clean_data(np.loadtxt("exp_data.txt"))
E_sif_raw, SIF_raw = clean_data(np.loadtxt("sif_data.txt"))

# ============================================================
# 3. 构造高精度卷积网格
# ============================================================
dE = 0.1
# 左右各延伸 20 eV，确保卷积核（SIF）完全进入数据区，防止边缘假象
PAD = 20.0
E_conv_grid = np.arange(E_LO - PAD, E_HI + PAD, dE)

# 插值 SIF (允许范围外填 0)
interp_sif = interp1d(
    E_sif_raw, SIF_raw, kind="linear", bounds_error=False, fill_value=0
)
SIF_conv_grid = interp_sif(E_conv_grid)
SIF_conv_grid = SIF_conv_grid / np.trapezoid(
    SIF_conv_grid, E_conv_grid
)  # SIF 面积归一化

# 截取实验数据拟合区间
mask_exp = (E_exp_raw >= E_LO) & (E_exp_raw <= E_HI)
E_exp_fit = E_exp_raw[mask_exp]
S_exp_fit = S_exp_raw[mask_exp]
S_exp_norm = S_exp_fit / np.trapezoid(S_exp_fit, E_exp_fit)

# ============================================================
# 4. 网格扫描并计算 RMS (扩大范围)
# ============================================================
# ★ 大幅扩大的扫描网格，确保等高线闭合
Te_grid = np.linspace(1, 30, 45)  # -5 ~ 30 eV
ne_grid = np.linspace(5e29, 20.0e29, 45)  # 0.5 ~ 10.0 x 10^23 cm^-3

RMS_matrix = np.zeros((len(ne_grid), len(Te_grid)))

print("开始扫描并计算 RMS (扩大范围版)...")
for i, ne_val in enumerate(ne_grid):
    for j, Te_val in enumerate(Te_grid):
        pc.Te_eV = Te_val
        pc.ne = ne_val
        pc.T = Te_val * pc.EV / pc.kB
        pc.wp = np.sqrt(ne_val * pc.e**2 / (pc.eps0 * pc.me))
        pc.kap = np.sqrt(ne_val * pc.e**2 / (pc.eps0 * pc.kB * pc.T))
        pc.mu = pc.chem_pot(pc.T, pc.ne)

        # 计算宽范围理论谱
        E_theo_raw, _, _, S_theo_raw, _ = pc.compute(
            THETA_DEG, E_LO - PAD, E_HI + PAD, nw=800, method="maxwell"
        )

        # 插值 + 卷积
        S_theo_grid = interp1d(
            E_theo_raw, S_theo_raw, kind="linear", bounds_error=False, fill_value=0
        )(E_conv_grid)
        S_theo_convolved = convolve(S_theo_grid, SIF_conv_grid, mode="same") * dE

        # 截取回拟合区间
        S_theo_final = interp1d(E_conv_grid, S_theo_convolved, kind="linear")(E_exp_fit)
        S_theo_norm = S_theo_final / np.trapezoid(S_theo_final, E_exp_fit)

        # RMS
        RMS_matrix[i, j] = np.sqrt(np.mean((S_theo_norm - S_exp_norm) ** 2))

print("计算完成！")

# ============================================================
# 5. 绘图：双图展示 (等高线 + 最佳拟合对比)
# ============================================================
NE_CM3_GRID = ne_grid / 1e29
TE_MESH, NE_MESH = np.meshgrid(Te_grid, NE_CM3_GRID)

# 寻找最小值
min_idx = np.unravel_index(np.argmin(RMS_matrix), RMS_matrix.shape)
min_Te = TE_MESH[min_idx]
min_ne = NE_MESH[min_idx]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ---- 图 1：RMS 等高线 ----
ax1 = axes[0]
contour = ax1.contourf(TE_MESH, NE_MESH, RMS_matrix, levels=40, cmap="viridis")
fig.colorbar(contour, ax=ax1, label="RMS Error")

# 画线并标注
cs = ax1.contour(
    TE_MESH, NE_MESH, RMS_matrix, levels=15, colors="k", linewidths=0.8, alpha=0.6
)
ax1.clabel(cs, inline=True, fontsize=9, fmt="%.4f")

ax1.plot(
    min_Te,
    min_ne,
    "r*",
    markersize=15,
    label=f"Min RMS\nTe={min_Te:.1f} eV\nne={min_ne:.2f}e23",
)
ax1.set_xlabel("Electron Temperature (eV)", fontsize=12)
ax1.set_ylabel(r"Electron Density (\(10^{23}\) cm\(^{-3}\))", fontsize=12)
ax1.set_title("RMS Contour (Wide Range)", fontsize=14)
ax1.legend(loc="upper right")

# ---- 图 2：最佳拟合对比谱 ----
ax2 = axes[1]

# 用最佳参数重新计算一次理论谱
pc.Te_eV = min_Te
pc.ne = min_ne * 1e29
pc.T = pc.Te_eV * pc.EV / pc.kB
pc.wp = np.sqrt(pc.ne * pc.e**2 / (pc.eps0 * pc.me))
pc.kap = np.sqrt(pc.ne * pc.e**2 / (pc.eps0 * pc.kB * pc.T))
pc.mu = pc.chem_pot(pc.T, pc.ne)

E_best_raw, _, _, S_best_raw, _ = pc.compute(
    THETA_DEG, E_LO - PAD, E_HI + PAD, nw=800, method="maxwell"
)
S_best_conv = (
    convolve(
        interp1d(E_best_raw, S_best_raw, bounds_error=False, fill_value=0)(E_conv_grid),
        SIF_conv_grid,
        mode="same",
    )
    * dE
)
S_best_norm = interp1d(E_conv_grid, S_best_conv)(E_exp_fit) / np.trapezoid(
    interp1d(E_conv_grid, S_best_conv)(E_exp_fit), E_exp_fit
)

ax2.plot(E_exp_fit, S_exp_norm, "g-", lw=2, label="Experiment (Normalized)")
ax2.plot(
    E_exp_fit,
    S_best_norm,
    "r--",
    lw=2,
    label=f"Best Fit (Convolved)\nTe={min_Te:.1f} eV, ne={min_ne:.2f}e23",
)
ax2.set_xlabel("Energy Shift (eV)", fontsize=12)
ax2.set_ylabel("Normalized Intensity", fontsize=12)
ax2.set_title("Best Fit vs Experiment (Compton Red Wing)", fontsize=14)
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
