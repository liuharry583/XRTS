"""XRTS：电子响应 -> 碰撞修正 -> Chihara 分解 -> 整谱仪器响应。

先修改“用户配置”。电子方法与碰撞模型是独立选择，详见各开关注释。
compute() 返回 (E, omega, free_raw, total_observed, k)，内部采用 SI。
未包含 bound-free 跃迁；历史低层接口保留，便于已有脚本调用。
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from functools import lru_cache, partial

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


# ============================================================
# 1. 用户配置：模型选择与参数
# ============================================================
Te_eV, ne, Zf, ZA = 33, 1.7e29, 3.3, 6.0  # 温度/eV, 密度/m^-3, 电离度, 原子序数
E0_keV = 4.75  # 探针能量/keV
# 电子响应：与碰撞模型独立选择。
# integral：有限温 Fermi-Dirac / Lindhard 积分，包含简并。
# maxwell ：非简并 Maxwell 分布的解析表达，需要 scipy。
# auto    ：旧策略，k/kappa>2 用 maxwell，否则 integral；不检验简并度。
ELECTRON_METHOD = "auto"

# 碰撞频率来源：
# none ：nu=0，无碰撞；忽略 NU_FRAC 和 nu_rad_s。
# const：nu=NU_FRAC*wp；显式配置 nu_rad_s 非 None 时优先（允许为 0）。
# bma  ：使用 nu_BMA(k, omega=wp) 的代表值，整条谱共用该值；
#        忽略 NU_FRAC/nu_rad_s，因此填 0 不会关闭 BMA(不影响）。
# BMA 内部固定采用 Lindhard 积分和 DH 屏蔽，不跟随 ELECTRON_METHOD/SII_MODE。
NU_MODE = "const"
NU_FRAC = 0.0

# 弹性项的离子静态结构因子：dh / arkhipov
SII_MODE = "dh"
INSTRUMENT_FWHM_eV = 42.0  # 整谱 FWHM [eV]，等效原标准差 18 eV
USE_CHIHARA_ION = True  # False 仅用于复现旧占位弹性权重，不是推荐物理模型


@dataclass(frozen=True)
class PhysicsConfig:
    """独立输入；默认值在导入时取自上方配置，批处理用 replace() 派生。

    运行中 configure() 更新全局入口，不会修改此类已确定的默认值。
    """

    Te_eV: float = Te_eV
    ne: float = ne
    Zf: float = Zf
    ZA: float = ZA
    E0_keV: float = E0_keV
    NU_MODE: str = NU_MODE
    SII_MODE: str = SII_MODE
    NU_FRAC: float = NU_FRAC
    USE_CHIHARA_ION: bool = USE_CHIHARA_ION
    nu_rad_s: float | None = None  # 仅 const 模式读取；优先于 NU_FRAC
    INSTRUMENT_FWHM_eV: float = INSTRUMENT_FWHM_eV
    ELECTRON_METHOD: str = ELECTRON_METHOD

    def __post_init__(self):
        for name in ("Te_eV", "ne", "Zf", "ZA", "E0_keV", "INSTRUMENT_FWHM_eV"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.NU_FRAC) or self.NU_FRAC < 0:
            raise ValueError("NU_FRAC must be finite and nonnegative")
        if self.nu_rad_s is not None and (
            not np.isfinite(self.nu_rad_s) or self.nu_rad_s < 0
        ):
            raise ValueError("nu_rad_s must be finite and nonnegative")
        choices = {
            "ELECTRON_METHOD": ("integral", "maxwell", "auto"),
            "NU_MODE": ("none", "const", "bma"),
            "SII_MODE": ("dh", "arkhipov"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")


# ---------- 2. 派生状态与兼容参数接口 ----------
@dataclass(frozen=True)
class DerivedState:
    config: PhysicsConfig
    T: float
    lam0: float
    wp: float
    kap: float
    nu: float  # const 输入值；实际采用值见 collision_frequency()
    mu: float


@lru_cache(maxsize=128)
def build_state(config):
    """只从完整输入推导；相同配置可复用，输入改变不会命中旧状态。"""
    temperature = config.Te_eV * EV / kB
    plasma = np.sqrt(config.ne * e**2 / (eps0 * me))
    return DerivedState(
        config=config,
        T=temperature,
        lam0=h * c / (config.E0_keV * 1e3 * EV),
        wp=plasma,
        kap=np.sqrt(config.ne * e**2 / (eps0 * kB * temperature)),
        nu=config.NU_FRAC * plasma if config.nu_rad_s is None else config.nu_rad_s,
        mu=chem_pot(temperature, config.ne),
    )


def current_config():
    """兼容旧调用方直接赋值基础参数；派生全局量不是输入。"""
    return PhysicsConfig(
        **{
            name: globals()[name]
            for name in PhysicsConfig.__dataclass_fields__
            if name != "nu_rad_s"
        }
    )


def refresh_state():
    """同步兼容接口的派生快照；compute/k_of_theta 会自动调用。"""
    state = build_state(current_config())
    globals().update(
        {name: getattr(state, name) for name in ("T", "lam0", "wp", "kap", "nu", "mu")}
    )
    return state


def configure(**changes):
    """一次更新基础参数并立即同步；如 configure(E0_keV=2.96)。"""
    if "nu_rad_s" in changes:
        raise ValueError("Use PhysicsConfig(nu_rad_s=...) with compute(config=...)")
    config = replace(current_config(), **changes)
    state = build_state(config)  # 验证和推导成功后才提交全局变更
    globals().update({name: getattr(config, name) for name in changes})
    globals().update(
        {name: getattr(state, name) for name in ("T", "lam0", "wp", "kap", "nu", "mu")}
    )
    return config


def k_of_theta(theta_deg, *, config=None):
    """散射角 → 散射波矢 k [m^-1]（论文 p66）。"""
    state = refresh_state() if config is None else build_state(config)
    if not np.isfinite(theta_deg) or not 0 < theta_deg <= 180:
        raise ValueError("theta_deg must be in (0, 180]")
    return 4 * np.pi * np.sin(np.deg2rad(theta_deg) / 2) / state.lam0


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
    这是积分正则化，不是碰撞频率或仪器宽度；精细计算需做网格收敛检查。
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
    """计算 nu_B(w) 的实部。w 是评估频率，不是手填的碰撞频率 nu。

    主流程只在 w=wp 评估一次，不是对整条谱逐频计算的完整动态 BMA。
    积分内部固定采用 Lindhard 与 DH 屏蔽。
    """
    w_safe = w + 1j * 1e-5 * (kB * T / hbar)
    q = np.linspace(1e-3 * k, 2 * k, 50)

    kappa2 = ne * e**2 / (eps0 * kB * T)
    integrand = np.zeros_like(q, dtype=complex)

    for i, qi in enumerate(q):
        eps_plus = epsilon_rpa(qi, w_safe, T, mu, **kw)
        eps_zero = epsilon_rpa(qi, 0.0, T, mu, **kw)
        V_ei2 = (Zf * e**2 / (eps0 * (qi**2 + kappa2))) ** 2
        sii = qi**2 / (qi**2 + kappa2)  # 这里用简化的 DH
        integrand[i] = qi**6 * V_ei2 * sii * (1 / w_safe) * (eps_plus - eps_zero)

    # BMA 符号修正：原文(张小丁式1.24)傅里叶约定与本文件 chi0 的 w+i·eta 约定
    # 相反，原式给出负 nu，此处前置因子取 -i 使 nu>0；与 physics_core.py 参考
    # 实现差一整体负号，属有意偏离，量级已经数值验证。
    nu = (
        (-1j * eps0 * ne) / (6 * np.pi**2 * e**2 * ne * me) * np.trapezoid(integrand, q)
    )
    return np.real(nu)


def _mermin_response(e_stat, e_dyn, w, nu):
    """两个电子分支共用的 Mermin 代数；零频由采样层处理。"""
    r = (e_dyn - 1) / (e_stat - 1)
    return 1 + (1 + 1j * nu / w) * (e_dyn - 1) / (1 + (1j * nu / w) * r)


def _structure_from_dielectric(eps, k, w, T, ne):
    """共用的涨落耗散系数；单电子归一，不强制重标谱面积。"""
    pre = (
        -hbar / (1 - np.exp(-hbar * w / (kB * T))) * (eps0 * k**2 / (np.pi * e**2 * ne))
    )
    return pre * np.imag(1 / eps)


# ---------- Maxwell 极限解析式 (§2.1 补充; 论文 p79) ----------
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
        return _mermin_response(e_stat, e_dyn, w, nu)

    def S_free_mw(k, w, T, nu, ne):
        eps = epsilon_mermin_mw(k, w, T, nu, ne)
        return _structure_from_dielectric(eps, k, w, T, ne)

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False  # 无 scipy 时跳过 Maxwell 解析式, 只用积分版


# ---------- Mermin (§2.2; 论文 (1.15) p86, Röpke–Selchow 形式) ----------
def epsilon_mermin(k, w, T, mu, nu, **kw):
    e_stat = epsilon_rpa(k, 0.0, T, mu, **kw)
    e_dyn = epsilon_rpa(k, w + 1j * nu, T, mu, **kw)
    return _mermin_response(e_stat, e_dyn, w, nu)


# ---------- 涨落耗散 → S^0_ee (§2.3; 论文 (1.11)_{3.2} p78 精确前置系数) ----------
def S_free(k, w, T, mu, nu, ne, **kw):
    """返回 S^0_ee(k,w) (单电子归一, 绝对单位, 不 rescale)。
    ∫S^0_ee dω = S^0_ee(k): 大 k→1, 小 k→k²/(k²+κ²) (屏蔽压低, 真实物理)"""
    eps = epsilon_mermin(k, w, T, mu, nu, **kw)
    return _structure_from_dielectric(eps, k, w, T, ne)


# ---------- 弹性峰 (§2.4; 论文 (1.5)_{3.1.2} p70, S_II(k,ω)≈S_II(k)δ(ω) p71) ----------
def f_I_k(k, ZA, z_scr=0.3):
    """离子形状因子, K 壳层 Pauling–Sherman 近似 (论文 p73)"""
    a_s = aB / (ZA - z_scr)
    return 1 / (1 + (k * a_s / 2) ** 2) ** 2


def S_elastic(w, k, ZA, Zf, width):
    """旧占位模型；Zf 只为兼容签名保留，权重 k->0 为 1。"""
    return f_I_k(k, ZA) ** 2 * gaussian_response(w, width)


# ---------- Chihara 离子项: f_i + rho, S_ii ----------
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
    rs = (3.0 / (4.0 * np.pi * ni)) ** (1 / 3) / aB
    TF = (hbar**2 / (2 * me * kB)) * (3 * np.pi**2 * ne) ** (2 / 3)
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
    A = (kB * Tg * np.log(2) * np.pi**1.5 * b ** (-1.5) * eps0) / e**2

    term1 = k**4
    term2 = (k**2 * kDe2) / (1 + k**2 * lam_ee**2)
    term3 = (k**2 * kDi2) / (1 + k**2 * lam_ii**2)
    term4 = (
        kDe2
        * kDi2
        * (
            1 / ((1 + k**2 * lam_ee**2) * (1 + k**2 * lam_ii**2))
            - 1 / (1 + k**2 * lam_ei**2) ** 2
        )
    )
    term5 = (
        A
        * k**2
        * kDe2
        * (k**2 + kDi2 / (1 + k**2 * lam_ii**2))
        * np.exp(-(k**2) / (4 * b))
    )
    Delta = term1 + term2 + term3 + term4 + term5

    bracket_1 = k**2 / (1 + k**2 * lam_ii**2)
    bracket_2 = kDe2 * (
        1 / ((1 + k**2 * lam_ee**2) * (1 + k**2 * lam_ii**2))
        - 1 / (1 + k**2 * lam_ei**2) ** 2
    )
    bracket_3 = A * k**2 * kDe2 / (1 + k**2 * lam_ii**2) * np.exp(-(k**2) / (4 * b))

    Phi_ii = (Zf * e**2 / (eps0 * Delta)) * (bracket_1 + bracket_2 + bracket_3)

    S_ii = 1.0 - (ni / (kB * Tg)) * Phi_ii
    return np.clip(S_ii, 1e-8, 1.0)


def S_ion_chihara(w, k, T, mu, ne, ZA, Zf, width, *, sii_mode=None, **kw):
    """Chihara 离子/弹性项：
       |f_i(k)+rho(k)|^2 * S_ii(k) * G(w; width)
    rho 和 S_ii 只算一次，不随 w 重复算。"""
    fi = f_i_core(k, ZA, Zf)
    rho = rho_cloud(k, T, mu, Zf, **kw)
    mode = SII_MODE if sii_mode is None else sii_mode
    if mode == "dh":
        sii = sii_dh(k, T, ne, Zf)
    elif mode == "arkhipov":
        sii = sii_arkhipov(k, T, ne, Zf, ZA)
    else:
        raise ValueError(f"Unknown SII_MODE: {mode!r}")
    weight = abs(fi + rho) ** 2 * sii
    return weight * gaussian_response(w, width)


# ---------- 6. 整谱仪器响应与卷积 ----------
def gaussian_response(w, sigma):
    """归一化响应 g(omega)，Glenzer & Redmer (2009) 式(53)，单位为 s。

    sigma 为角频率标准差 [rad/s]；能量标准差需乘 e/hbar。
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    return np.exp(-0.5 * (np.asarray(w) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def convolve_instrument(w, w_source, spectrum, sigma):
    """式(54)：在频移坐标上积分 S(w') g(w-w') dw'。

    显式梯形权重保留 dω 与谱密度单位，支持非均匀源网格。
    源区间外按零处理；调用者须扩展源区间以避免显示窗口截断效应。
    不按输出窗口重新归一化：有限窗口外的面积是真实的流出。
    """
    w = np.asarray(w, dtype=float)
    w_source = np.asarray(w_source, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if (
        w.ndim != 1
        or w_source.ndim != 1
        or w_source.size < 2
        or spectrum.shape != w_source.shape
        or not np.isfinite(w).all()
        or not np.isfinite(w_source).all()
        or not np.isfinite(spectrum).all()
        or not np.all(np.diff(w_source) > 0)
    ):
        raise ValueError("Expected finite 1D arrays and increasing source grid")
    # 同时验证 sigma，即使输出数组为空。
    gaussian_response(0.0, sigma)
    steps = np.diff(w_source)
    weights = np.empty_like(w_source)
    weights[0], weights[-1] = steps[0] / 2, steps[-1] / 2
    weights[1:-1] = (steps[:-1] + steps[1:]) / 2
    weighted = spectrum * weights
    result = np.empty_like(w)
    # 分块，避免输出网格 × 源网格矩阵占用过多内存。
    for start in range(0, w.size, 64):
        stop = start + 64
        kernel = gaussian_response(w[start:stop, None] - w_source, sigma)
        result[start:stop] = kernel @ weighted
    return result


# ---------- 7. 模型选择：各开关只在这里分派 ----------
def resolve_electron_method(k, state, method=None):
    """显式 method 优先；否则读取配置。auto 仅按几何阈值选择。"""
    selected = state.config.ELECTRON_METHOD if method is None else method
    if selected == "auto":
        selected = "maxwell" if k / state.kap > 2.0 else "integral"
    if selected not in ("integral", "maxwell"):
        raise ValueError("method must be integral, maxwell, or auto")
    if selected == "maxwell" and not HAS_SCIPY:
        raise RuntimeError("Maxwell 解析式需要 scipy；可显式选择 integral")
    return selected


def collision_frequency(k, state):
    """实际采用的 nu [rad/s]。BMA 为 w=wp 单点近似，非逐频动态计算。

    none -> 0；const -> nu_rad_s（非 None）或 NU_FRAC*wp；
    bma -> nu_BMA(k, wp)，此时忽略 NU_FRAC 和 nu_rad_s。
    """
    cfg = state.config
    if cfg.NU_MODE == "none":
        return 0.0
    if cfg.NU_MODE == "const":
        return state.nu
    return nu_BMA(k, state.wp, state.T, cfg.ne, cfg.Zf, state.mu)


def instrument_sigma(config):
    """唯一宽度输入 FWHM [eV] -> 高斯标准差 [rad/s]。"""
    return config.INSTRUMENT_FWHM_eV / np.sqrt(8 * np.log(2)) * e / hbar


def _source_frequency_grid(E_lo, E_hi, nw, sigma, wp):
    """源区间扩展 ±8 sigma；至少分辨 sigma/6 与 0.01 wp。"""
    output_step = (E_hi - E_lo) * e / hbar / (nw - 1)
    source_step_max = min(sigma / 6, 0.01 * wp)
    subdivisions = max(1, int(np.ceil(output_step / source_step_max)))
    step = output_step / subdivisions
    padding = int(np.ceil(8 * sigma / step))
    return (
        -E_hi * e / hbar
        + np.arange(-padding, (nw - 1) * subdivisions + padding + 1) * step
    )


def _sample_source(evaluate, w_source, wp):
    """用两侧插值处理 Mermin/FDT 在零频的可去奇点，不删除积分点。"""
    cutoff = 1e-3 * wp
    near_zero = np.abs(w_source) <= cutoff
    source = np.empty_like(w_source)
    source[~near_zero] = [evaluate(wi) for wi in w_source[~near_zero]]
    if near_zero.any():
        source[near_zero] = np.interp(
            w_source[near_zero],
            [-cutoff, cutoff],
            [evaluate(-cutoff), evaluate(cutoff)],
        )
    return source


# ---------- 8. 计算与绘图入口 ----------
def normalize_spectrum(spectrum, reference_peak=None):
    """按给定峰高归一化；未给定时使用当前谱的最大值。

    峰高须为有限正数。空谱、零谱不产生伪造的归一化结果。
    """
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.ndim != 1 or not spectrum.size or not np.isfinite(spectrum).all():
        raise ValueError("Expected a nonempty finite 1D spectrum")
    peak = np.max(spectrum) if reference_peak is None else reference_peak
    if not np.isscalar(peak) or not np.isfinite(peak) or peak <= 0:
        raise ValueError("Normalization reference peak must be finite and positive")
    return spectrum / peak


def compute(theta_deg, E_lo, E_hi, nw=600, method=None, *, config=None, normalize=True):
    """E_lo/E_hi 为光子能量位移 [eV]，下移为负。

    返回 E（降序）、w（升序）、未展宽自由电子谱 S_free、展宽总谱 S_tot、k。
    默认 normalize=True：S_free 和 S_tot 分别除以输出区间内自身最大值，
    因此两条返回谱的最大值各为 1。
    两条返回谱分别归一化，不再共用绝对单位，不能直接相加减。
    normalize=False 保留原绝对谱与单电子归一约定，供物理积分/拟合使用。
    method 显式指定时覆盖 ELECTRON_METHOD；省略则使用配置。
    显式 config 不影响全局状态；省略时使用模块基础参数。
    保留旧输出网格的近零频排除约定；卷积源网格单独补齐零频邻域。
    """
    state = refresh_state() if config is None else build_state(config)
    cfg = state.config
    T, wp, mu = state.T, state.wp, state.mu
    ne, Zf, ZA = cfg.ne, cfg.Zf, cfg.ZA
    if (
        not np.isfinite([E_lo, E_hi]).all()
        or E_lo >= E_hi
        or isinstance(nw, (bool, np.bool_))
        or not isinstance(nw, (int, np.integer))
        or nw < 2
    ):
        raise ValueError("Invalid energy grid")
    k = k_of_theta(theta_deg, config=cfg)
    E = np.linspace(E_lo, E_hi, nw)
    w = -E * e / hbar  # 光子位移 → omega（能量损失为正的约定）
    m = np.abs(w) > 1e-3 * wp  # 避开 ω=0 处 FDT 因子的 0/0
    E, w = E[m], w[m]
    idx = np.argsort(w)  # w 升序：trapezoid 积分要求 x 递增
    E, w = E[idx], w[idx]
    selected = resolve_electron_method(k, state, method)
    nu_eff = collision_frequency(k, state)
    if selected == "maxwell":
        evaluate = partial(S_free_mw, k, T=T, nu=nu_eff, ne=ne)
    else:
        evaluate = partial(S_free, k, T=T, mu=mu, nu=nu_eff, ne=ne)
    S = np.array([evaluate(wi) for wi in w])
    sigma = instrument_sigma(cfg)
    w_source = _source_frequency_grid(E_lo, E_hi, nw, sigma, wp)
    source = _sample_source(evaluate, w_source, wp)
    S_observed = convolve_instrument(w, w_source, source, sigma)
    # 离子本征项为 A(k) delta(omega)，与同一响应卷积后解析地等于
    # A(k) g(omega; sigma)。直接计算此解析结果，不先造宽峰再重复卷积。
    if cfg.USE_CHIHARA_ION:
        S_ion = S_ion_chihara(
            w,
            k,
            T,
            mu,
            ne,
            ZA,
            Zf,
            width=sigma,
            sii_mode=cfg.SII_MODE,
        )
    else:
        S_ion = S_elastic(
            w,
            k,
            ZA,
            Zf,
            width=sigma,
        )
    S_tot = Zf * S_observed + S_ion
    if normalize:
        S = normalize_spectrum(S)
        S_tot = normalize_spectrum(S_tot)
    return E, w, S, S_tot, k


def plot_spectrum(
    theta_deg, E_lo, E_hi, nw=600, method=None, *,
    config=None, spectrum="S_tot", normalize=True,
):
    """选择 S_free 或 S_tot 绘图；默认按各自参考峰归一化。"""
    if spectrum not in ("S_free", "S_tot"):
        raise ValueError("spectrum must be S_free or S_tot")
    cfg = current_config() if config is None else config
    E, _, free, total, k = compute(
        theta_deg, E_lo, E_hi, nw, method, config=cfg, normalize=normalize
    )
    state = build_state(cfg)
    selected = resolve_electron_method(k, state, method)
    plt.figure()
    plt.plot(E, free if spectrum == "S_free" else total, "k-", lw=1.2)
    plt.axvline(0, color="gray", ls="--", lw=0.8)
    plt.xlabel("Photon energy shift (eV)")
    plt.ylabel("Normalized intensity" if normalize else "S(k,ω) (arb. units)")
    plt.title(
        f"θ={theta_deg}°, α={state.kap / k:.2f}; {selected}, {cfg.NU_MODE}; "
        f"Te={cfg.Te_eV:g}eV, ne={cfg.ne:.0e}/m³, Zf={cfg.Zf:g}"
    )
    plt.xlim(E_lo, E_hi)
    plt.ylim(0, None)
    plt.show()


refresh_state()


if __name__ == "__main__":
    plot_spectrum(120.0, -250, 100)  # 非集体：弹性峰 + Compton峰
    plot_spectrum(5.0, -45, 45)  # 集体：弹性峰 + plasmon对峰
