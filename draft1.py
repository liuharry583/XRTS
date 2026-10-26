import numpy as np
import matplotlib.pyplot as plt

# ---------- 常数 (SI; 论文全章 SI，未用原子单位) ----------
hbar = 1.054571817e-34
kB = 1.380649e-23
me = 9.1093837015e-31
e = 1.602176634e-19
eps0 = 8.8541878128e-12
c = 2.99792458e8
h = 2 * np.pi * hbar
aB = 5.29177210903e-11
EV = e  # 1 eV in J


# ---------- 化学势: n = 2*(2π m kT/h^2)^{3/2} F_{1/2}(βμ), 二分反解 ----------
def fermi_half(eta, xmax=60.0, n=4000):
    x = np.linspace(0, xmax, n)
    f = np.sqrt(x) / (np.exp(np.clip(x - eta, -700, 700)) + 1.0)
    return (2 / np.sqrt(np.pi)) * np.trapezoid(f, x)


def chem_pot(T, ne):
    """T: K, ne: m^-3 → μ (J)"""
    pre = 2 * (2 * np.pi * me * kB * T / h**2) ** 1.5
    target = ne / pre
    lo, hi = -50.0, 200.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if fermi_half(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * kB * T


# ---------- 有限温 Lindhard χ0 (§2.1; 论文 (1.11)_{3.2.1} p79) ----------
# 约定: p = 动量(kg m/s), k = 波矢(m^-1); 标准推迟极化率带整体负号
# 与论文形式代数等价(已验算), 只是把 1/ħ 吸收进 χ0 定义(论文 p84 亦如此)
def chi0(k, w, T, mu, eta_rel=1e-3, np_grid=160, nmu=80):
    """k: m^-1 (标量), w: rad/s (可复数), mu: 化学势 (J)"""
    beta = 1 / (kB * T)
    pmax = np.sqrt(2 * me * max(10 * kB * T + mu, 10 * kB * T)) * 4
    p = np.linspace(0, pmax, np_grid)
    mang = np.linspace(-1, 1, nmu)
    P, M = np.meshgrid(p, mang, indexing="ij")
    hk = hbar * k
    e_plus = (P**2 + hk**2 / 4 + P * hk * M) / (2 * me)
    e_minus = (P**2 + hk**2 / 4 - P * hk * M) / (2 * me)
    f = lambda ee: 1 / (np.exp(np.clip(beta * (ee - mu), -700, 700)) + 1.0)
    num = f(e_minus) - f(e_plus)  # = -(f+ - f-): 推迟极化率约定
    w_c = w + 1j * eta_rel * (kB * T / hbar)  # i0+ → 有限展宽
    den = hbar * w_c - P * hk * M / me
    inner = np.trapezoid(num / den, mang, axis=1)
    return (1 / (2 * np.pi**2 * hbar**3)) * np.trapezoid(p**2 * inner, p)


def epsilon_rpa(k, w, T, mu, **kw):
    return 1 - (e**2 / (eps0 * k**2)) * chi0(k, w, T, mu, **kw)


# ---------- Maxwell 极限解析式 (§2.1 补充; 论文 p79) ----------
# 用途: (1) 非简并 (η≪-1) 时与积分版 chi0 交叉验证; (2) 大 k 非集体区——积分版在共振 μ*
# 处很尖, 实测 480×240 网格仍有 ~10% 数值抖动, 解析式无噪声且快 ~20 倍。
# 注意 η≈-1.4 有轻度简并, 解析式与积分版差百分之几~十几%, 属预期内, 不是 bug。
try:
    from scipy.special import wofz

    def _W(x):
        """W(x) = 1 + x·Z(x), Z 为等离子体色散函数 (论文 p79)"""
        return 1 + x * 1j * np.sqrt(np.pi) * wofz(x)

    def epsilon_rpa_maxwell(k, w, T, ne):
        """k: m^-1, w: rad/s (可复数), T: K, ne: m^-3"""
        vt = np.sqrt(kB * T / me)
        wp = np.sqrt(ne * e**2 / (eps0 * me))
        xe = w / (np.sqrt(2) * k * vt)
        kap = hbar * k / (2 * np.sqrt(2) * me * vt)
        t1 = (1 - _W(xe + kap)) / (xe + kap)
        t2 = (1 - _W(xe - kap)) / (xe - kap)
        return 1 + (wp**2 / (k**2 * vt**2)) * (1 / (4 * kap)) * (t1 - t2)

    def epsilon_mermin_mw(k, w, T, nu, ne):
        e_stat = epsilon_rpa_maxwell(k, 0.0, T, ne)
        e_dyn = epsilon_rpa_maxwell(k, w + 1j * nu, T, ne)
        r = (e_dyn - 1) / (e_stat - 1)
        return 1 + (1 + 1j * nu / w) * (e_dyn - 1) / (1 + (1j * nu / w) * r)

    def S_free_mw(k, w, T, nu, ne):
        eps = epsilon_mermin_mw(k, w, T, nu, ne)
        pre = (
            -hbar
            / (1 - np.exp(-hbar * w / (kB * T)))
            * (eps0 * k**2 / (np.pi * e**2 * ne))
        )
        return pre * np.imag(1 / eps)

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False  # 无 scipy 时跳过 Maxwell 解析式, 只用积分版


# ---------- Mermin (§2.2; 论文 (1.15) p86, Röpke–Selchow 形式) ----------
def epsilon_mermin(k, w, T, mu, nu, **kw):
    e_stat = epsilon_rpa(k, 0.0, T, mu, **kw)
    e_dyn = epsilon_rpa(k, w + 1j * nu, T, mu, **kw)
    r = (e_dyn - 1) / (e_stat - 1)
    return 1 + (1 + 1j * nu / w) * (e_dyn - 1) / (1 + (1j * nu / w) * r)


# ---------- 涨落耗散 → S^0_ee (§2.3; 论文 (1.11)_{3.2} p78 精确前置系数) ----------
def S_free(k, w, T, mu, nu, ne, **kw):
    """返回 S^0_ee(k,w) (单电子归一, 绝对单位, 不 rescale)。
    ∫S^0_ee dω = S^0_ee(k): 大 k→1, 小 k→k²/(k²+κ²) (屏蔽压低, 真实物理)"""
    eps = epsilon_mermin(k, w, T, mu, nu, **kw)
    pre = (
        -hbar / (1 - np.exp(-hbar * w / (kB * T))) * (eps0 * k**2 / (np.pi * e**2 * ne))
    )
    return pre * np.imag(1 / eps)


# ---------- 弹性峰 (§2.4; 论文 (1.5)_{3.1.2} p70, S_II(k,ω)≈S_II(k)δ(ω) p71) ----------
def f_I_k(k, ZA, z_scr=0.3):
    """离子形状因子, K 壳层 Pauling–Sherman 近似 (论文 p73)"""
    a_s = aB / (ZA - z_scr)
    return 1 / (1 + (k * a_s / 2) ** 2) ** 2


def S_elastic(w, k, ZA, Zf, width):
    weight = f_I_k(k, ZA) ** 2  # |f_I(k)|²·S_II(k), S_II(k)≈1 占位; k→0 回到 (ZA-Zf)²
    return weight * np.exp(-0.5 * (w / width) ** 2) / (width * np.sqrt(2 * np.pi))


# ---------- demo: 跑通 + 自检 ----------
# 用两个散射角分别演示集体区(等离激元)与非集体区(Compton 反冲)
if __name__ == "__main__":
    Te_eV, ne, Zf, ZA = 20.0, 1e29, 2.0, 4.0  # 示例参数(类 Be)
    T = Te_eV * EV / kB
    E0_keV = 5.0
    lam0 = h * c / (E0_keV * 1e3 * EV)
    wp = np.sqrt(ne * e**2 / (eps0 * me))
    kap = np.sqrt(ne * e**2 / (eps0 * kB * T))
    nu = 0.05 * wp  # 常数碰撞频率(论文 p95: 0<ω<3ωp 内 ν 基本平坦, 常数近似误差小);
    # 取 0.05ωp 使等离激元峰宽 ~ 网格可分辨 (自检4要求 Δω ≲ ν/3)
    mu = chem_pot(T, ne)

    def spectrum(theta_deg, w_lo, w_hi, nw=800):
        k = 4 * np.pi * np.sin(np.deg2rad(theta_deg) / 2) / lam0
        w = np.linspace(w_lo * wp, w_hi * wp, nw)
        w = w[np.abs(w) > 1e-3 * wp]  # 避开 ω=0 处 FDT 因子的 0/0
        S = np.array([S_free(k, wi, T, mu, nu, ne) for wi in w])
        return k, w, S  # 绝对单位, 不 rescale

    # (a) 前向小角: 集体区 —— 自检2/3/4 都在这里验
    k, w, S = spectrum(5.0, -2.0, 2.0)
    area = np.trapezoid(S, w)
    stat = (k / kap) ** 2 / (1 + (k / kap) ** 2)
    print(
        f"[θ=5°] k = {k / kap:.2f}κ_D (α={kap / k:.1f}): "
        f"非弹性峰位 ≈ {w[np.argmax(S)] / wp:.2f} ωp  (自检2: k→0 时应→1)"
    )
    print(
        f"∫S^0_ee dω = {area:.4f}, 期望 k²/(k²+κ²) = {stat:.4f}  (自检4: 屏蔽压低是物理)"
    )
    i = np.argmin(np.abs(w - wp))
    j = np.argmin(np.abs(w + wp))
    print(
        f"细致平衡 S(-ωp)/S(ωp) = {S[j] / S[i]:.4f}, 理论 e^(-ħωp/kT) = {np.exp(-hbar * wp / (kB * T)):.4f}"
    )
    # (b) 背向大角: 非集体区 —— 用 Maxwell 解析式 (积分版在此 k 下需极密 μ 网格, 见数值陷阱注)
    # 峰应在 Compton 反冲位移 ħ²k²/2me 附近, 面积→1
    if HAS_SCIPY:
        k = 4 * np.pi * np.sin(np.deg2rad(120.0) / 2) / lam0
        w = np.linspace(0.5 * wp, 12 * wp, 1200)
        w = w[w > 1e-3 * wp]
        S = np.array([S_free_mw(k, wi, T, 1e14, ne) for wi in w])
        area = np.trapezoid(S, w)
        recoil = (hbar * k) ** 2 / (2 * me) / (hbar * wp)
        print(
            f"[θ=120°] k = {k / kap:.2f}κ_D (α={kap / k:.2f}): "
            f"非弹性峰位 ≈ {w[np.argmax(S)] / wp:.2f} ωp  (反冲位移 = {recoil:.2f} ωp)"
        )
        print(
            f"∫S^0_ee dω = {area:.4f}  (自检4: 非集体极限应→1; 差值来自有限 ω 范围截断的尾巴)"
        )
    else:
        print("[θ=120°] 跳过 (需 scipy.special.wofz)")


def plot_spectrum(theta_deg, e_lo, e_hi, nw=600):
    k = 4 * np.pi * np.sin(np.deg2rad(theta_deg) / 2) / lam0
    w = np.linspace(e_lo * e / hbar, e_hi * e / hbar, nw)
    w = w[np.abs(w) > 1e-3 * wp]
    S = np.array([S_free(k, wi, T, mu, nu, ne) for wi in w])
    S_tot = Zf * S + S_elastic(w, k, ZA, Zf, width=2.0 * e / hbar)
    E_shift = -hbar * w / e  # 光子能量位移，下移为负（论文惯例）
    plt.plot(E_shift, S_tot)
    plt.axvline(0, color="gray", ls="--")
    plt.xlabel("Photon energy shift (eV)")
    plt.ylabel("S(k,ω) (arb. units)")
    plt.show()


plot_spectrum(120.0, -250, 100)  # 非集体：弹性峰 + Compton峰
plot_spectrum(5.0, -45, 45)  # 集体：弹性峰 + plasmon对峰
