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
amu = 1.66053906660e-27  # kg; physics_core.py 的 sii_arkhipov 使用的离子质量单位


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
def chi0(k, w, T, mu, eta_scale=0.05, np_grid=160, nmu=80):
    """k: m^-1 (标量), w: rad/s (可复数), mu: 化学势 (J)

    eta_scale: i0+ 展宽取为 eta_scale * (p_th*ħk/m), p_th=√(2mkT)。
    自适应 (k,T) 的原因: 展宽必须同时满足
      (a) 物理上 ≪ 特征尺度 (静态响应、朗道宽度), 否则抹平物理
          (固定 1e-3*kT 在 k 小/T 高时把 ε(k,0) 从 1+α² 压到 ~1, 等离激元峰消失);
      (b) 数值上 ≳ μ 网格分辨, 否则极点比网格还尖, 积分变噪声
          (固定 1e-5*kT 在 T=0.5eV 时极点宽度仅为网格的 1/4000, 谱成毛刺)。
    取 Doppler 尺度的 5%: 极点被约 2 个 μ 网格点分辨, 物理扭曲 <5%。
    旧版 eta_rel (相对 kT) 已废弃, 如需手动控制直接传 eta_scale。
    """
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
    p_th = np.sqrt(2 * me * kB * T)
    eta_abs = eta_scale * (p_th * hbar * k / me)  # 自适应展宽 (J)
    w_c = w + 1j * eta_abs / hbar
    den = hbar * w_c - P * hk * M / me
    inner = np.trapezoid(num / den, mang, axis=1)
    return (1 / (2 * np.pi**2 * hbar**3)) * np.trapezoid(p**2 * inner, p)


def epsilon_rpa(k, w, T, mu, **kw):
    return 1 - (e**2 / (eps0 * k**2)) * chi0(k, w, T, mu, **kw)


# ---------- BMA：移植自 physics_core.py 的 nu_BMA ----------
# 单位一致：k/q: m^-1, w/nu: rad/s, T: K, ne: m^-3, mu: J。
# 内部 DH 屏蔽与 DH sii 属于 BMA 公式本身，不受 SII_MODE 影响。
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

    # BMA 符号修正：原文(张小丁式1.24)傅里叶约定与本文件 chi0 的 w+i·eta 约定
    # 相反，原式给出负 nu，此处前置因子取 -i 使 nu>0；与 physics_core.py 参考
    # 实现差一整体负号，属有意偏离，量级已经数值验证。
    nu = (-1j * eps0 * ne) / (6 * np.pi**2 * e**2 * ne * me) * np.trapezoid(integrand, q)
    return np.real(nu)


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


# ---------- Chihara 离子项: f_i + rho, S_ii (新增) ----------
def f_i_core(k, ZA, Zf, z_scr=0.3):
    """核/紧束缚电子形状因子，k->0 返回 Z_A-Z_f。
    注意：原 f_I_k(k) 的 k->0 返回 1，不能直接当 f_i 用。"""
    Zc = max(ZA - Zf, 0.0)
    a_s = aB / max(ZA - z_scr, 1e-12)
    return Zc / (1.0 + (k * a_s / 2.0) ** 2) ** 2


def rho_cloud(k, T, mu, Zf, **kw):
    """Chihara 屏蔽云形状因子，第一版线性屏蔽近似：
       rho(k) = Zf * [1 - 1/eps_RPA(k,0)]
    k->0 -> Zf, k->inf -> 0。
    论文未给显式，采用 Chihara 2000 / Gregori 2003 线性屏蔽云形式。
    """
    if k < 1e-12:
        return Zf
    eps = np.real(epsilon_rpa(k, 0.0, T, mu, **kw))
    eps = max(eps, 1.0 + 1e-12)  # 防数值噪声导致 1/eps 异常
    return Zf * (1.0 - 1.0 / eps)


def sii_dh(k, T, ne, Zf, Ti=None):
    """DH/OCP 离子静态结构因子：S_ii = k^2/(k^2+kappa_i^2)。
    k->0 -> 0, k->inf -> 1。论文表 3.1 的 DH 形式。"""
    Ti = T if Ti is None else Ti
    if Zf <= 0:
        return 1.0
    ni = ne / Zf
    kappa_i2 = Zf**2 * ni * e**2 / (eps0 * kB * Ti)
    return k**2 / (k**2 + kappa_i2)


# ---------- Arkhipov：移植自 physics_core.py 的 sii_arkhipov ----------
# 单位一致：k: m^-1, T/TF/Tq/Tg: K, ne/ni: m^-3, 热波长: m。
# 保留参考实现 mi = ZA * amu 和 rs 使用 ni 的约定；无需 eV/SI 换算。
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


def S_ion_chihara(w, k, T, mu, ne, ZA, Zf, width, **kw):
    """Chihara 离子/弹性项：
       |f_i(k)+rho(k)|^2 * S_ii(k) * G(w; width)
    rho 和 S_ii 只算一次，不随 w 重复算。"""
    fi = f_i_core(k, ZA, Zf)
    rho = rho_cloud(k, T, mu, Zf, **kw)
    if SII_MODE == "dh":
        sii = sii_dh(k, T, ne, Zf)
    elif SII_MODE == "arkhipov":
        sii = sii_arkhipov(k, T, ne, Zf, ZA)
    else:
        raise ValueError(f"Unknown SII_MODE: {SII_MODE!r}")
    weight = abs(fi + rho) ** 2 * sii
    return weight * np.exp(-0.5 * (w / width) ** 2) / (width * np.sqrt(2 * np.pi))


# ============================================================
# ★ 参数设置：只改这里一处，下面所有计算自动跟随
# ============================================================
Te_eV, ne, Zf, ZA = 33, 1.7e29, 3.3, 6.0  # 温度/eV, 密度/m^-3, 电离度, 原子序数
E0_keV = 4.75  # 探针能量/keV
NU_MODE = "const"  # "const": 模块级 nu；"bma": 每次 compute 的局部碰撞频率
SII_MODE = "dh"  # "dh" 或 "arkhipov"；外部跑批脚本可覆盖
NU_FRAC = (
    0.0  # 碰撞频率 nu = NU_FRAC * wp（论文p95：0<ω<3ωp内ν基本平坦，常数近似误差小）
)
ELASTIC_WIDTH_eV = 18  # 弹性峰宽度占位（实际按仪器分辨定）
USE_CHIHARA_ION = True  # True: 用 |f_i+rho|^2*S_ii 弹性项; False: 回退原 S_elastic

# ---------- 派生量（自动算，不用手改） ----------
T = Te_eV * EV / kB
lam0 = h * c / (E0_keV * 1e3 * EV)
wp = np.sqrt(ne * e**2 / (eps0 * me))
kap = np.sqrt(ne * e**2 / (eps0 * kB * T))
nu = NU_FRAC * wp
mu = chem_pot(T, ne)


def k_of_theta(theta_deg):
    """散射角 → 散射波矢 k（论文 p66，未编号公式）。原来散在三处手写，现统一"""
    return 4 * np.pi * np.sin(np.deg2rad(theta_deg) / 2) / lam0


def compute(theta_deg, E_lo, E_hi, nw=600, method="auto"):
    """核心计算（画图和自检共用，避免重复写网格逻辑）。
    E_lo/E_hi: 光子能量位移范围[eV]，下移为负（论文惯例），直接按想看的范围填。
    返回 E[eV](升序), w[rad/s](升序), S_free(单电子归一), S_tot(含弹性峰), k。
    method='auto': k/kap>2（非集体）自动用 Maxwell 解析式，避开积分版大k数值噪声；
              'integral'/'maxwell' 可强制指定。
    """
    k = k_of_theta(theta_deg)
    E = np.linspace(E_lo, E_hi, nw)
    w = -E * e / hbar  # 光子位移 → omega（能量损失为正的约定）
    m = np.abs(w) > 1e-3 * wp  # 避开 ω=0 处 FDT 因子的 0/0
    E, w = E[m], w[m]
    idx = np.argsort(w)  # w 升序：trapezoid 积分要求 x 递增
    E, w = E[idx], w[idx]
    nu_eff = nu  # const 原样读取模块级 nu，不重新计算或修改它
    if NU_MODE == "bma":
        # 对称能量窗下 mean(w)=0，恰落在 nu_BMA 的 w=0 奇点上；
        # 实测 nu 在 0.1~2 倍 wp 间变化平缓（<25%），取等离子体频率处的值作代表。
        w_rep = wp
        nu_bma = nu_BMA(k, w_rep, T, ne, Zf, mu)
        nu_eff = nu_bma
    elif NU_MODE != "const":
        raise ValueError(f"Unknown NU_MODE: {NU_MODE!r}")
    use_mw = (k / kap > 2.0) if method == "auto" else (method == "maxwell")
    if use_mw:
        if not HAS_SCIPY:
            raise RuntimeError("Maxwell 解析式需要 scipy")
        S = np.array([S_free_mw(k, wi, T, nu_eff, ne) for wi in w])
    else:
        S = np.array([S_free(k, wi, T, mu, nu_eff, ne) for wi in w])
    # ---- 离子/弹性项：Chihara 或原占位 ----
    if USE_CHIHARA_ION:
        S_ion = S_ion_chihara(
            w, k, T, mu, ne, ZA, Zf,
            width=ELASTIC_WIDTH_eV * e / hbar,
        )
    else:
        S_ion = S_elastic(
            w, k, ZA, Zf,
            width=ELASTIC_WIDTH_eV * e / hbar,
        )
    S_tot = Zf * S + S_ion
    return E, w, S, S_tot, k


def plot_spectrum(theta_deg, E_lo, E_hi, nw=600, method="auto"):
    """画一条 S–energy shift 谱线（论文 Fig.3.2 风格）"""
    E, w, S, S_tot, k = compute(theta_deg, E_lo, E_hi, nw, method)
    plt.figure()
    plt.plot(E, S_tot, "k-", lw=1.2)
    plt.axvline(0, color="gray", ls="--", lw=0.8)
    plt.xlabel("Photon energy shift (eV)")
    plt.ylabel("S(k,ω) (arb. units)")
    plt.title(
        f"θ={theta_deg}°, α={kap / k:.2f} (Te={Te_eV:g}eV, ne={ne:.0e}/m³, Zf={Zf:g})"
    )
    plt.xlim(E_lo, E_hi)
    plt.ylim(0, None)
    plt.show()


def run_selfchecks():
    """四个自检（诊断用）：静态屏蔽 / 峰位 / 细致平衡 / 求和规则"""
    print("=== 自检 ===")
    # (a) θ=5° 集体区
    E, w, S, _, k = compute(5.0, -45, 45, nw=400, method="integral")
    eV = hbar * wp / e
    print(
        f"[θ=5°] α={kap / k:.1f}: 非弹性峰位≈{abs(E[np.argmax(S)]) / eV:.2f}ωp（自检2: k→0应→1）"
    )
    print(
        f"  ∫S⁰dω={np.trapezoid(S, w):.4f}, "
        f"期望k²/(k²+κ²)={(k / kap) ** 2 / (1 + (k / kap) ** 2):.4f}（自检4: 屏蔽压低是物理）"
    )
    i = np.argmin(np.abs(w - wp))
    j = np.argmin(np.abs(w + wp))
    print(
        f"  细致平衡 S(-ωp)/S(+ωp)={S[j] / S[i]:.4f}, "
        f"理论={np.exp(-hbar * wp / (kB * T)):.4f}（自检3）"
    )
    # (b) θ=120° 非集体区（解析版，无噪声）
    E, w, S, _, k = compute(120.0, -250, 100, nw=600, method="maxwell")
    recoil = (hbar * k) ** 2 / (2 * me) / (hbar * wp)
    print(
        f"[θ=120°] α={kap / k:.2f}: 非弹性峰位≈{abs(E[np.argmax(S)]) / eV:.2f}ωp "
        f"（反冲={recoil:.2f}ωp，自检2）"
    )
    print(f"  ∫S⁰dω={np.trapezoid(S, w):.4f}（自检4: 应→1，差值为尾巴截断）")


if __name__ == "__main__":
    run_selfchecks()
    plot_spectrum(120.0, -250, 100)  # 非集体：弹性峰 + Compton峰
    plot_spectrum(5.0, -45, 45)  # 集体：弹性峰 + plasmon对峰