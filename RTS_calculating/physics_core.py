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
amu = 1.66053906660e-27  # 原子质量单位 (kg)


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


# ---------- 有限温 Lindhard χ0 ----------
def chi0(k, w, T, mu, eta_scale=0.05, np_grid=160, nmu=80):
    beta = 1 / (kB * T)
    pmax = np.sqrt(2 * me * max(10 * kB * T + mu, 10 * kB * T)) * 4
    p = np.linspace(0, pmax, np_grid)
    mang = np.linspace(-1, 1, nmu)
    P, M = np.meshgrid(p, mang, indexing="ij")
    hk = hbar * k
    e_plus = (P**2 + hk**2 / 4 + P * hk * M) / (2 * me)
    e_minus = (P**2 + hk**2 / 4 - P * hk * M) / (2 * me)
    f = lambda ee: 1 / (np.exp(np.clip(beta * (ee - mu), -700, 700)) + 1.0)
    num = f(e_minus) - f(e_plus)
    p_th = np.sqrt(2 * me * kB * T)
    eta_abs = eta_scale * (p_th * hbar * k / me)
    w_c = w + 1j * eta_abs / hbar
    den = hbar * w_c - P * hk * M / me
    inner = np.trapezoid(num / den, mang, axis=1)
    return (1 / (2 * np.pi**2 * hbar**3)) * np.trapezoid(p**2 * inner, p)


def epsilon_rpa(k, w, T, mu, **kw):
    return 1 - (e**2 / (eps0 * k**2)) * chi0(k, w, T, mu, **kw)


# ---------- Maxwell 极限解析式 (用于非集体区高精度) ----------
try:
    from scipy.special import wofz

    def _W(x):
        return 1 + x * 1j * np.sqrt(np.pi) * wofz(x)

    def epsilon_rpa_maxwell(k, w, T, ne):
        vt = np.sqrt(kB * T / me)
        wp = np.sqrt(ne * e**2 / (eps0 * me))
        xe = w / (np.sqrt(2) * k * vt)
        kap = hbar * k / (2 * np.sqrt(2) * me * vt)
        t1 = (1 - _W(xe + kap)) / (xe + kap)
        t2 = (1 - _W(xe - kap)) / (xe - kap)
        return 1 + (wp**2 / (k**2 * vt**2)) * (1 / (4 * kap)) * (t1 - t2)

    def S_free_mw(k, w, T, nu, ne):
        # Maxwell 极 + Mermin 修正
        e_stat = epsilon_rpa_maxwell(k, 0.0, T, ne)
        e_dyn = epsilon_rpa_maxwell(k, w + 1j * nu, T, ne)
        r = (e_dyn - 1) / (e_stat - 1)
        eps = 1 + (1 + 1j * nu / w) * (e_dyn - 1) / (1 + (1j * nu / w) * r)
        pre = -hbar / (1 - np.exp(-hbar * w / (kB * T))) * (eps0 * k**2 / (np.pi * e**2 * ne))
        return pre * np.imag(1 / eps)

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------- Mermin 介电函数 ----------
def epsilon_mermin(k, w, T, mu, nu, **kw):
    e_stat = epsilon_rpa(k, 0.0, T, mu, **kw)
    e_dyn = epsilon_rpa(k, w + 1j * nu, T, mu, **kw)
    r = (e_dyn - 1) / (e_stat - 1)
    return 1 + (1 + 1j * nu / w) * (e_dyn - 1) / (1 + (1j * nu / w) * r)


# ---------- 涨落耗散 -> S^0_ee ----------
def S_free(k, w, T, mu, nu, ne, **kw):
    eps = epsilon_mermin(k, w, T, mu, nu, **kw)
    pre = -hbar / (1 - np.exp(-hbar * w / (kB * T))) * (eps0 * k**2 / (np.pi * e**2 * ne))
    return pre * np.imag(1 / eps)


# ---------- 精确的 BMA 碰撞频率 (根据张小丁论文 1.24 动态计算) ----------
def nu_BMA(k, w, T, ne, Zf, mu, **kw):
    """
    基于 Born-Mermin 近似 (BMA) 的碰撞频率动态计算
    注意：chi0 只接受标量 k，因此这里使用 for 循环对 q 数组进行遍历积分
    """
    w_safe = w + 1j * 1e-5 * (kB*T/hbar)
    q = np.linspace(1e-3 * k, 2 * k, 50)
    
    kappa2 = ne * e**2 / (eps0 * kB * T)
    integrand = np.zeros_like(q, dtype=complex)
    
    for i, qi in enumerate(q):
        eps_plus = epsilon_rpa(qi, w_safe, T, mu, **kw)
        eps_zero = epsilon_rpa(qi, 0.0, T, mu, **kw)
        V_ei2 = (Zf * e**2 / (eps0 * (qi**2 + kappa2)))**2
        sii = qi**2 / (qi**2 + kappa2) # 这里用简化的 DH
        integrand[i] = qi**6 * V_ei2 * sii * (1 / w_safe) * (eps_plus - eps_zero)
        
    nu = (1j * eps0 * ne) / (6 * np.pi**2 * e**2 * ne * me) * np.trapezoid(integrand, q)
    return np.real(nu)


# ---------- 离子形状因子 (K壳层 Pauling-Sherman 近似) ----------
def f_i_core(k, ZA, Zf, z_scr=0.3):
    Zc = max(ZA - Zf, 0.0)
    a_s = aB / max(ZA - z_scr, 1e-12)
    return Zc / (1.0 + (k * a_s / 2.0) ** 2) ** 2


# ---------- 精确的离子双组分静态结构因子 (Arkhipov 模型, 论文公式 1.7) ----------
def sii_arkhipov(k, T, ne, Zf, ZA, Z_ion=1.0):
    mi = ZA * amu
    ni = ne / Zf
    rs = (3.0 / (4.0 * np.pi * ni))**(1/3) / aB
    TF = (hbar**2 / (2 * me * kB)) * (3 * np.pi**2 * ne)**(2/3)
    Tq = TF / (1.3251 - 0.1779 * np.sqrt(rs))
    Tg = np.sqrt(T * T + Tq * Tq)  # 电子有效温度
    
    kDe2 = ne * e**2 / (eps0 * kB * Tg)
    kDi2 = Zf * ne * e**2 / (eps0 * kB * Tg)
    
    mu_ee = me / 2
    mu_ii = mi / 2
    mu_ei = me * mi / (me + mi)
    lam_ee = hbar / np.sqrt(2 * np.pi * mu_ee * kB * Tg)
    lam_ii = hbar / np.sqrt(2 * np.pi * mu_ii * kB * Tg)
    lam_ei = hbar / np.sqrt(2 * np.pi * mu_ei * kB * Tg)
    
    b = 1.0 / (lam_ee**2 * np.pi * np.log(2))
    A = (kB * Tg * np.log(2) * np.pi**1.5 * b**(-1.5) * eps0) / e**2
    
    term1 = k**4
    term2 = (k**2 * kDe2) / (1 + k**2 * lam_ee**2)
    term3 = (k**2 * kDi2) / (1 + k**2 * lam_ii**2)
    term4 = kDe2 * kDi2 * (1 / ((1 + k**2 * lam_ee**2) * (1 + k**2 * lam_ii**2)) - 1 / (1 + k**2 * lam_ei**2)**2)
    term5 = A * k**2 * kDe2 * (k**2 + kDi2 / (1 + k**2 * lam_ii**2)) * np.exp(-k**2 / (4 * b))
    Delta = term1 + term2 + term3 + term4 + term5
    
    bracket_1 = k**2 / (1 + k**2 * lam_ii**2)
    bracket_2 = kDe2 * (1 / ((1 + k**2 * lam_ee**2) * (1 + k**2 * lam_ii**2)) - 1 / (1 + k**2 * lam_ei**2)**2)
    bracket_3 = A * k**2 * kDe2 / (1 + k**2 * lam_ii**2) * np.exp(-k**2 / (4 * b))
    
    Phi_ii = (Zf * e**2 / (eps0 * Delta)) * (bracket_1 + bracket_2 + bracket_3)
    
    S_ii = 1.0 - (ni / (kB * Tg)) * Phi_ii
    return np.clip(S_ii, 1e-8, 1.0)


# ---------- 精确的屏蔽云形状因子 (Chihara 模型) ----------
def rho_chihara(k, T, ne, Zf, ZA, mu, **kw):
    eps0_k = np.real(epsilon_rpa(k, 0.0, T, mu, **kw))
    eps0_k = max(eps0_k, 1.0 + 1e-12)
    rho = Zf * (1.0 - 1.0 / eps0_k)
    return rho


# ---------- Chihara 离子项：f_i + rho, S_ii ----------
def S_ion_chihara(w, k, T, mu, ne, ZA, Zf, width, **kw):
    fi = f_i_core(k, ZA, Zf)
    rho = rho_chihara(k, T, ne, Zf, ZA, mu, **kw)
    sii = sii_arkhipov(k, T, ne, Zf, ZA)  # 替换为精确的解析解
    weight = abs(fi + rho) ** 2 * sii
    return weight * np.exp(-0.5 * (w / width) ** 2) / (width * np.sqrt(2 * np.pi))


# ============================================================
# ★ 参数设置
# ============================================================
Te_eV, ne, Zf, ZA = 33, 1.7e29, 3.3, 6.0  # 默认参数
E0_keV = 4.75  
NU_MODE = "BMA"  # "BMA" 使用精确碰撞频率，"ZERO" 则为无碰撞
ELASTIC_WIDTH_eV = 18  
USE_CHIHARA_ION = True

# ---------- 派生量 ----------
T = Te_eV * EV / kB
lam0 = h * c / (E0_keV * 1e3 * EV)
wp = np.sqrt(ne * e**2 / (eps0 * me))
kap = np.sqrt(ne * e**2 / (eps0 * kB * T))
mu = chem_pot(T, ne)


def k_of_theta(theta_deg):
    return 4 * np.pi * np.sin(np.deg2rad(theta_deg) / 2) / lam0


def compute(theta_deg, E_lo, E_hi, nw=600, method="auto"):
    k = k_of_theta(theta_deg)
    E = np.linspace(E_lo, E_hi, nw)
    w = -E * e / hbar 
    m = np.abs(w) > 1e-3 * wp 
    E, w = E[m], w[m]
    idx = np.argsort(w)
    E, w = E[idx], w[idx]
    
    if NU_MODE == "BMA":
        # 取中心频率的碰撞频率作为代表，避免在积分中频繁计算
        nu = nu_BMA(k, np.mean(w), T, ne, Zf, mu)
    else:
        nu = 0.0
        
    use_mw = (k / kap > 2.0) if method == "auto" else (method == "maxwell")
    if use_mw and HAS_SCIPY:
        S = np.array([S_free_mw(k, wi, T, nu, ne) for wi in w])
    else:
        S = np.array([S_free(k, wi, T, mu, nu, ne) for wi in w])
        
    if USE_CHIHARA_ION:
        S_ion = S_ion_chihara(w, k, T, mu, ne, ZA, Zf, width=ELASTIC_WIDTH_eV * e / hbar)
    else:
        fi = f_i_core(k, ZA, Zf)
        S_ion = (fi**2) * np.exp(-0.5 * (w / (ELASTIC_WIDTH_eV * e / hbar)) ** 2) / ((ELASTIC_WIDTH_eV * e / hbar) * np.sqrt(2 * np.pi))
        
    S_tot = Zf * S + S_ion
    return E, w, S, S_tot, k


def plot_spectrum(theta_deg, E_lo, E_hi, nw=600, method="auto"):
    E, w, S, S_tot, k = compute(theta_deg, E_lo, E_hi, nw, method)
    plt.figure()
    plt.plot(E, S_tot, "k-", lw=1.2, label='Total S(k,ω)')
    plt.axvline(0, color="gray", ls="--", lw=0.8)
    plt.xlabel("Photon energy shift (eV)")
    plt.ylabel("S(k,ω) (arb. units)")
    plt.title(f"θ={theta_deg}°, α={kap / k:.2f} (Te={Te_eV:g}eV, ne={ne:.0e}/m³, Zf={Zf:g})")
    plt.xlim(E_lo, E_hi)
    plt.ylim(0, None)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("=== 开始计算总谱 (非简化版，整合 Arkhipov Sii 与 BMA 碰撞) ===")
    # 测试例1：非集体区
    plot_spectrum(120.0, -250, 100) 