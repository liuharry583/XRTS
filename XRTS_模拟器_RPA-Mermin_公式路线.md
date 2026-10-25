# XRTS 谱线模拟器：RPA + Mermin 公式路线与代码脚手架

> 背景：10/12 早组会，老师布置下周内（约 10/19 前）写一个"造数据"小程序：
> 输入 $(T_e, n_e, Z_f)$ → 输出一条模拟 XRTS 谱线。方法：照中文博士论文 p61–95 的简化模型，
> **RPA + Mermin（含碰撞项）**，先别碰 DFT/深度学习，bound-free 可先跳过。
>
> **2026-10-13 晚更新**：论文（中科大博士论文第 3 章《动力学形状因子的几种理论模型与计算结果》，
> 印刷页 61–95，扫描版）已到手并逐页核对。本文件所有公式均已对上论文编号，符号/单位制/SI 与论文一致。
> 论文公式编号按小节重起（如 (1.11) 在 3.2 与 3.2.1 指不同公式），下文引用时连小节一起标注。
> 配套提取清单：`thesis_ch3_formulas.md`（同目录，35 页逐页提取，410 行）。

## 1. 计算路线总览

```
输入: T_e, n_e, Z_f (+ 探针能量 E0, 散射角 θs, 碰撞频率 ν)
  ↓  散射几何: k = 4π sin(θs/2)/λ0,  λ0 = hc/E0        [论文 p66, 未编号]
χ0(k,ω): 有限温 Lindhard 极化率 (§2.1)                  [论文 (1.11)_{3.2.1}, p79]
  ↓  ε_RPA = 1 − V_k χ0,   V_k = e²/(ε0 k²)
ε_Mermin(k,ω): 碰撞修正 (§2.2)                          [论文 (1.15), p86, Röpke–Selchow 形式]
  ↓  涨落耗散定理 (§2.3)                                [论文 (1.11)_{3.2}, p78]
S_ee(k,ω): 自由电子动态结构因子（精确前置系数）→ 求和规则验算 ∫S dω ≈ Z_f
  ↓  叠弹性峰 (§2.4)                                    [论文 (1.5)_{3.2.1→3.1.2}, p70 Chihara 分解]
S_total(k,ω) = S_弹性(ω≈0) + Z_f·S^0_ee   (+ bound-free 以后再补，对应第三项)
```

## 2. 关键公式

### 2.1 有限温 Lindhard 极化率 χ₀ —— 论文 (1.11)_{3.2.1} (p79)

论文原文写的是介电函数（$\mathbf{p}$ 取**动量**，$\mathbf{k}$ 取**波矢**，
测度 $2d^3p/(2\pi\hbar)^3$ 中 2 为自旋简并，$f$ 为 Fermi–Dirac 分布且**归一到 1**，
$n_e$ 不显式出现——与化学势反解式配合使用）：

$$\varepsilon^{\mathrm{RPA}}(\mathbf{k},\omega)=1-\frac{e^2}{\hbar\varepsilon_0k^2}\int\frac{f(\mathbf{p}+\hbar\mathbf{k}/2)-f(\mathbf{p}-\hbar\mathbf{k}/2)}{\mathbf{k}\!\cdot\!\mathbf{p}/m_e-\omega-i\nu}\,\frac{2d^3p}{(2\pi\hbar)^3},\qquad \nu\to0^+$$

下面 χ₀ 写法与上式**代数等价**（已验算：把分母乘 $-\hbar$ 即得标准推迟形式），
只是把 $1/\hbar$ 吸收进 χ₀ 定义（论文 p84 的 3.2.1.2 也是这么写的）：

$$\chi_0(k,\omega)=-\frac{1}{2\pi^2\hbar^3}\int_0^\infty p^2dp\int_{-1}^{1}d\mu\,
\frac{f(\varepsilon_+)-f(\varepsilon_-)}{\hbar\omega-p\hbar k\mu/m_e+i0^+}$$

其中 $\varepsilon_\pm=(p^2+\hbar^2k^2/4\pm p\hbar k\mu)/(2m_e)$，
$f(\varepsilon)=1/(e^{\beta(\varepsilon-\mu)}+1)$，$\beta=1/k_BT_e$。
负号是标准推迟极化率的约定（保证 $\mathrm{Im}\,\chi_0<0\ (\omega>0)$，
从而 $\mathrm{Im}\,\varepsilon>0$ 为物理耗散；静态 $\chi_0(k,0)=-\partial n/\partial\mu<0$
给出 $\varepsilon(k,0)=1+V_k\partial n/\partial\mu>1$ 即正常屏蔽）。

化学势 $\mu(T,n_e)$ 由下式二分反解（论文未给更直接的取法，保持现状）：

$$n_e = 2\left(\frac{2\pi m_e k_BT_e}{h^2}\right)^{3/2} F_{1/2}(\beta\mu),
\qquad F_{1/2}(\eta)=\frac{2}{\sqrt\pi}\int_0^\infty\frac{\sqrt{x}\,dx}{e^{x-\eta}+1}$$

数值实现要点：把 $i0^+$ 写成有限小量 $w \to w + i\eta$，
取 $\eta \sim 10^{-3}\,k_BT_e/\hbar$（热展宽量级），直接传复数频率即可。

> 提速备选（论文 p79/p82）：Maxwell 极限有解析式
> $\varepsilon=1+\frac{\omega_p^2}{k^2v_t^2}\frac{1}{4\kappa}[\frac{1-W(x_e+\kappa)}{x_e+\kappa}-\frac{1-W(x_e-\kappa)}{x_e-\kappa}]$
> （$W$ 为 Faddeeva 型函数）；Arista–Brandt 简化 (1.12) 把 χ₀ 化为对 $g(x)$ 的一维积分。
> 非简并调试阶段可用解析式交叉验证。

### 2.2 Mermin 碰撞修正 —— 论文 (1.15) (p86)

论文原式（已对扫描件核对；注意是 $\varepsilon^M-1=+\frac{\cdots}{\cdots}$，
即下式，**不是** $1-$）：

$$\varepsilon^M(k,\omega)=1+
\frac{(1+i\nu(\omega)/\omega)\,[\varepsilon^{\mathrm{RPA}}(k,\omega+i\nu(\omega))-1]}
{1+(i\nu(\omega)/\omega)\,[\varepsilon^{\mathrm{RPA}}(k,\omega+i\nu(\omega))-1]/[\varepsilon^{\mathrm{RPA}}(k,0)-1]}$$

- $\nu(\omega)$：**动力学碰撞频率，明确含频**。论文实际程序采用的是 (1.24) (p91)：
  $\nu^{B}(\omega)=\nu^{LB}(\omega)$（Lenard–Balescu），
  $V_\alpha(k)=Z_fe^2/[\varepsilon_0(k^2+\kappa_{sc}^2)]$，
  配表 3.1 的屏蔽模型（TCS：$S_i=1$；DH：$S_i=k^2/(k^2+\kappa_s^2)$；HNC：数值解）。
  这是后续升级项，先记下位置。
- **p95 关键实用结论**：$0<\omega<3\omega_{pe}$ 内 ν 较高且基本平坦，
  $\omega>3\omega_{pe}$ 后迅速降低、对谱贡献可忽略；
  "拟合时可用**固定碰撞频率**近似代替动力学碰撞频率，误差很小"。
  → 调试阶段设常数 $\nu\sim0.1\,\omega_p$ 是论文背书的做法，不是凑合。
- 注意 $\varepsilon^{\mathrm{RPA}}(k,\omega+i\nu)$ 就是把 §2.1 的 $\omega$ 换成复数直接算；
  分母另含静态值 $\varepsilon^{\mathrm{RPA}}(k,0)$。

### 2.3 涨落耗散定理 → S(k,ω) —— 论文 (1.11)_{3.2} (p78)

论文给出**精确前置系数**（之前"先不用纠结"的地方，现在有了）：

$$S^0_{ee}(k,\omega)=
\underbrace{-\frac{\hbar}{1-e^{-\hbar\omega/k_BT_e}}}_{\text{细致平衡因子}}
\;\times\;
\underbrace{\frac{\varepsilon_0k^2}{\pi e^2n_e}}_{\text{单位换算}}
\;\times\;
\underbrace{\mathrm{Im}\!\left[\frac{1}{\varepsilon^M(k,\omega)}\right]}_{\text{损耗函数（}\omega>0\text{ 时为负）}}$$

整体为正。$\hbar\omega\ll k_BT_e$ 时分母 $\to\hbar\omega/k_BT_e$，谱对称；
$T_e$ 低时分母导致强烈不对称（论文 p78 原文讨论）——这正是细致平衡
$S(k,-\omega)=e^{-\hbar\omega/k_BT_e}S(k,\omega)$ [论文 (1.6), p70] 的来源。

**面积的物理（含自检升级）**：$\int_{-\infty}^{\infty}S^0_{ee}(k,\omega)d\omega=S^0_{ee}(k)$
为自由电子**静态结构因子**（单电子归一）：非集体极限（$k$ 大）$\to1$；
小 $k$ 处被屏蔽压低，经典极限下 $S^0_{ee}(k)\approx k^2/(k^2+\kappa^2)$。
**不要**把面积强行 rescale 到 $Z_f$——$Z_f$ 因子在 Chihara 分解中本就写在外面
（第二项是 $Z_fS^0_{ee}$），小 $k$ 处的压低是真实的屏蔽物理，rescale 会把它错误放大。
数值积分后把面积与期望的静态值对比（大 $k$ 比 1，小 $k$ 比 $k^2/(k^2+\kappa^2)$），
差得多先查网格是否分辨了峰（等离激元峰宽 $\sim\nu$，要求 $\Delta\omega\lesssim\nu/3$），再查实现。

### 2.4 弹性峰 Rayleigh（ω≈0）—— 论文 (1.5)_{3.1.2} (p70) Chihara 分解

$$S^{\mathrm{tot}}_{ee}(\mathbf{k},\omega)=\underbrace{|f_I(\mathbf{k})+\rho(\mathbf{k})|^2S_{II}(\mathbf{k},\omega)}_{\text{弹性项}}+\underbrace{Z_fS^0_{ee}(\mathbf{k},\omega)}_{\text{自由电子项}}+\underbrace{Z_B\!\int\!\tilde{S}_{ce}(\mathbf{k},\omega-\omega')S_I(\mathbf{k},\omega')d\omega'}_{\text{bound-free（暂跳过，老师允许）}}$$

- $S_{II}(\mathbf{k},\omega)\approx S_{II}(\mathbf{k})\,\delta(\omega)$（论文 p71：
  离子振荡频率约电子的 1/40，探测器分辨不出）→ 数值上用窄高斯占位，宽度取离子热运动/仪器分辨。
- 权重取 $|f_I(k)|^2S_{II}(k)$，屏蔽云 $\rho(k)$ 先略去；
  离子形状因子用 K 壳层 Pauling–Sherman 近似（论文 p73）：
  $$f_{I,1s}(k)=\frac{1}{[1+(ka_s/2)^2]^2},\qquad a_s=\frac{a_B}{Z_A-0.3}$$
  $k\to0$ 时 $|f_I|^2\to(Z_A-Z_f)^2$，回到之前最简取法的极限，自洽。
- $S_{II}(k)$ 可用 Arhipov–Davletov 公式 (1.7) (p71–72)，第一版先取 $S_{II}(k)\approx1$ 占位。

Salpeter模型、RPA模型、QCSA模型