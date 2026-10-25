import numpy as np
import matplotlib.pyplot as plt
import RTS计算.physics_core as pc

# ============================================================
# ★★★ 新增：Matplotlib 渲染配置 ★★★
# ============================================================
plt.rcParams["mathtext.fontset"] = "cm"  # 使用类似 LaTeX 的 Computer Modern 数学字体
plt.rcParams["font.family"] = "sans-serif"  # 正文使用无衬线字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# ============================================================

# ============================================================
# 严格使用 Kritcher 2011 论文中的物理参数 (Be靶, 9 keV探针, 135°)
# ============================================================
pc.ZA = 4.0  # 铍 (Beryllium)
pc.Zf = 2.0  # 平均电离度 (Be+2)
pc.E0_keV = 9.0  # Zn He-alpha 探针能量
THETA_DEG = 135.0  # 散射角

# 拟合能区：Kri 原文的 Compton 下移红翼部分
# 9keV 探针反冲约260eV，所以重点看 -400 到 -50 eV 左右的形状
E_LO = -400.0
E_HI = -50.0
NW_POINTS = 200

# 设定为 0 碰撞（非集体散射区对碰撞频率不敏感，原文近似如此）
pc.NU_MODE = "ZERO"


def update_pc_params(Te_eV, ne):
    """仅更新 physics_core 内的物理状态，不做任何额外操作"""
    pc.Te_eV = Te_eV
    pc.ne = ne
    pc.T = Te_eV * pc.EV / pc.kB
    pc.wp = np.sqrt(ne * pc.e**2 / (pc.eps0 * pc.me))
    pc.kap = np.sqrt(ne * pc.e**2 / (pc.eps0 * pc.kB * pc.T))
    pc.mu = pc.chem_pot(pc.T, ne)


# ============================================================
# 1. 生成基准“实验数据” (使用 Kri 原文 Fig 3 的最优拟合值)
# ============================================================
TRUE_TE = 14.0  # eV
TRUE_NE = 1.1e30  # m^-3 (1.1e24 cm^-3)

print("生成基准谱 (True Te=14eV, ne=1.1e24/cc)...")
update_pc_params(TRUE_TE, TRUE_NE)
E_grid, _, _, S_exp, _ = pc.compute(
    THETA_DEG, E_LO, E_HI, nw=NW_POINTS, method="maxwell"
)

# ============================================================
# 2. 严格按原文进行网格扫描，计算绝对 RMS 值
# ============================================================
# 设定扫描范围 (依据Kri Fig 3的坐标轴，稍微提高分辨率让等高线更平滑)
Te_grid = np.linspace(5, 20, 50)  # 5 ~ 20 eV
ne_grid = np.linspace(0.5e30, 2.0e30, 50)  # 0.5 ~ 2.0 x 10^24 cm^-3

RMS_matrix = np.zeros((len(ne_grid), len(Te_grid)))

print("开始扫描并计算未归一化的绝对 RMS...")
for i, ne_val in enumerate(ne_grid):
    for j, Te_val in enumerate(Te_grid):
        update_pc_params(Te_val, ne_val)

        # 使用相同参数配置生成当前点的理论谱
        _, _, _, S_theo, _ = pc.compute(
            THETA_DEG, E_LO, E_HI, nw=NW_POINTS, method="maxwell"
        )

        # 直接计算原始差值的 RMS (不归一化，不缩放)
        rms_val = np.sqrt(np.mean((S_theo - S_exp) ** 2))
        RMS_matrix[i, j] = rms_val

print("计算完成，绘制原汁原味的等高线图...")

# ============================================================
# 3. 绘制等高线图 (完美 LaTeX 风格)
# ============================================================
NE_CM3_GRID = ne_grid / 1e30
TE_MESH, NE_MESH = np.meshgrid(Te_grid, NE_CM3_GRID)

plt.figure(figsize=(7, 6))

# 使用标准的 viridis 色图
contour = plt.contourf(TE_MESH, NE_MESH, RMS_matrix, levels=25, cmap="viridis")
cbar = plt.colorbar(contour)
cbar.set_label("RMS", fontsize=12)  # 颜色条标签

# 画一些细细的黑色等高线作为辅助
plt.contour(
    TE_MESH, NE_MESH, RMS_matrix, levels=25, colors="k", linewidths=0.3, alpha=0.3
)

# 找最小值并标记
min_idx = np.unravel_index(np.argmin(RMS_matrix), RMS_matrix.shape)
min_Te = TE_MESH[min_idx]
min_ne = NE_MESH[min_idx]

plt.plot(
    min_Te,
    min_ne,
    "r*",
    markersize=12,
    label=rf"Min RMS at: \(T_e={min_Te:.1f}\ \mathrm{{eV}}, n_e={min_ne:.2f}\times 10^{{24}}\ \mathrm{{cm}}^{{-3}}\)",
)

# 坐标轴设置 (使用 raw string 和 $ 语法)
plt.xlabel(r"Electron Temperature \(T_e\) (eV)", fontsize=14)
plt.ylabel(r"Electron Density \(n_e\) (\(10^{24}\ \mathrm{cm}^{-3}\))", fontsize=14)

# 图标题 (展示完美的 \theta 和 ^\circ 上标)
plt.title(r"RMS Contour (Kritcher Method, \(\theta=135.0^\circ\))", fontsize=15, pad=10)

# 设置图例字体大小
plt.legend(loc="upper right", fontsize=11)

plt.tight_layout()
plt.show()
