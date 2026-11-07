"""Sakurai 中文第3版 §3.6：球谐函数、概率测度与轨道旋转的教学补充。

依赖 NumPy、Matplotlib；直接运行将在本目录生成两张 PNG/SVG 和核验 JSON。
采用教材的 Condon–Shortley 相位约定，矩阵检查取 ħ=1。
图像来自解析模型，不是实验数据；本节原题未要求数值程序。
这份复算材料不代表学习者已经独立作答或掌握。
"""
from pathlib import Path
from math import factorial, pi
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
FONT = Path("C:/Windows/Fonts/msyh.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#f8fafc",
    "axes.facecolor": "white",
})
BLUE, ORANGE, GREEN = "#2867a8", "#d27437", "#238577"
PURPLE, LIGHT = "#8162a5", "#dce4ed"
CHECKS = {}
RNG = np.random.default_rng(20261024)


def record(name, error, tolerance=2e-11):
    error = float(np.max(np.abs(error)))
    error = max(error, CHECKS.get(name, {}).get("max_abs_error", 0.0))
    CHECKS[name] = {
        "max_abs_error": error,
        "tolerance": tolerance,
        "passed": bool(error <= tolerance),
    }
    if error > tolerance:
        raise AssertionError((name, error, tolerance))


def associated_legendre(l, m, x):
    """P_l^m(x), m>=0，含 Condon–Shortley 的 (-1)^m。"""
    x = np.asarray(x)
    pmm = np.ones_like(x, dtype=float)
    for k in range(1, m + 1):
        pmm *= -(2 * k - 1) * np.sqrt(np.maximum(0.0, 1 - x * x))
    if l == m:
        return pmm
    pm1 = (2 * m + 1) * x * pmm
    if l == m + 1:
        return pm1
    p0, p1 = pmm, pm1
    for k in range(m + 2, l + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k + m - 1) * p0) / (k - m)
    return p1


def ylm(l, m, theta, phi):
    """归一球谐 Y_l^m(theta,phi)，theta 的标准域为 [0,pi]。"""
    if m < 0:
        return (-1) ** (-m) * np.conjugate(ylm(l, -m, theta, phi))
    normalization = np.sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return normalization * associated_legendre(l, m, np.cos(theta)) * np.exp(1j * m * phi)


def angular(l):
    m = np.arange(l, -l - 1, -1, dtype=float)
    raising = np.zeros((len(m), len(m)), complex)
    for column in range(1, len(m)):
        raising[column - 1, column] = np.sqrt((l - m[column]) * (l + m[column] + 1))
    lowering = raising.conj().T
    return m, (raising + lowering) / 2, (raising - lowering) / (2j), np.diag(m)


def rotation(generator, angle):
    values, vectors = np.linalg.eigh(generator)
    return (vectors * np.exp(-1j * angle * values)) @ vectors.conj().T


def sphere_grid():
    u, weights = np.polynomial.legendre.leggauss(80)
    phi = np.linspace(0, 2 * pi, 128, endpoint=False)
    theta = np.arccos(u)[:, None]
    phi = phi[None, :]
    # dOmega = dphi du，无需再乘 sin(theta)。
    weights = weights[:, None] * (2 * pi / phi.shape[1]) * np.ones_like(phi)
    return theta, phi, weights


def numerical_checks():
    theta, phi, weights = sphere_grid()
    labels = [(l, m) for l in range(6) for m in range(-l, l + 1)]
    basis = np.array([ylm(l, m, theta, phi) * np.ones_like(weights) for l, m in labels])
    flat = basis.reshape(len(labels), -1)
    gram = (flat.conj() * weights.ravel()) @ flat.T
    record("sphere_orthonormality_l_0_to_5", gram - np.eye(len(labels)))

    for l in range(6):
        values = np.array([ylm(l, m, theta, phi) for m in range(-l, l + 1)])
        record("addition_theorem_same_direction",
               (np.abs(values) ** 2).sum(axis=0) - (2 * l + 1) / (4 * pi))
        mvals, lx, ly, lz = angular(l)
        for _ in range(20):
            alpha = RNG.uniform(0, 2 * pi)
            beta = RNG.uniform(0, pi)
            d_column = rotation(ly, beta)[:, l]
            d_column = np.exp(-1j * mvals * alpha) * d_column
            expected = np.sqrt(4 * pi / (2 * l + 1)) * np.array([
                np.conjugate(ylm(l, int(m), beta, alpha)) for m in mvals
            ])
            record("spherical_harmonic_D_column", d_column - expected)
            record("d_00_equals_legendre",
                   rotation(ly, beta)[l, l] - associated_legendre(l, 0, np.cos(beta)))

    # 角概率边缘分布：对 theta 的密度必须包含球面测度中的 sin(theta)。
    angles = np.linspace(0, pi, 40001)
    for l, m in [(0, 0), (1, 0), (1, 1)]:
        p_theta = 2 * pi * np.sin(angles) * np.abs(ylm(l, m, angles, 0.0)) ** 2
        record("polar_marginal_normalization", np.trapezoid(p_theta, angles) - 1, 2e-8)
    px = np.sqrt(3 / (4 * pi)) * np.sin(theta) * np.cos(phi)
    p_phi_px = (np.abs(px) ** 2 * weights).sum(axis=0) / (2 * pi / phi.shape[1])
    record("real_px_azimuth_marginal", p_phi_px - np.cos(phi.ravel()) ** 2 / pi)
    y11 = ylm(1, 1, theta, phi)
    p_phi_y11 = (np.abs(y11) ** 2 * weights).sum(axis=0) / (2 * pi / phi.shape[1])
    record("complex_Y11_azimuth_marginal", p_phi_y11 - 1 / (2 * pi))

    # 旋转后的实 p 轨道：ψ_a(n)=sqrt(3/4pi) a·n；它是 L_a 本征值0的态。
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta) * np.ones_like(phi)
    l1_basis = np.array([ylm(1, m, theta, phi) for m in (1, 0, -1)])
    _, lx, ly, lz = angular(1)
    for beta in np.linspace(-2 * pi, 2 * pi, 65):
        axis = np.array([np.sin(beta), 0.0, np.cos(beta)])
        psi = np.sqrt(3 / (4 * pi)) * (axis[0] * nx + axis[2] * nz)
        coefficients = np.sum(l1_basis.conj() * psi[None, :, :] * weights[None, :, :], axis=(1, 2))
        expected = rotation(ly, beta)[:, 1]
        record("rotated_real_p_spherical_coefficients", coefficients - expected)
        record("rotated_real_p_probabilities", abs(coefficients) ** 2 -
               np.array([np.sin(beta) ** 2 / 2, np.cos(beta) ** 2, np.sin(beta) ** 2 / 2]))
        record("rotated_real_p_axis_eigenvalue_zero", (axis[0] * lx + axis[2] * lz) @ coefficients)
        record("rotated_real_p_all_mean_L_zero",
               [np.vdot(coefficients, a @ coefficients) for a in (lx, ly, lz)])
        record("rotated_real_p_norm", np.sum(weights * abs(psi) ** 2) - 1)
        record("rotated_real_p_L_squared", np.vdot(coefficients,
               (lx @ lx + ly @ ly + lz @ lz) @ coefficients) - 2)

    # 原题3.23角函数：直接球面积分独立提取系数，验证相位与概率。
    psi = np.sqrt(3 / (44 * pi)) * (nx + ny + 3 * nz)
    coefficients = np.sum(l1_basis.conj() * psi[None, :, :] * weights[None, :, :], axis=(1, 2))
    target = np.array([(-1 + 1j) / np.sqrt(22), 3 / np.sqrt(11), (1 + 1j) / np.sqrt(22)])
    record("exercise_3_23_coefficient_phases", coefficients - target)
    record("exercise_3_23_probabilities", abs(coefficients) ** 2 - np.array([1, 9, 1]) / 11)

    # 3.26为前节已完成题，本节用球谐函数再检查，不能增加原题覆盖计数。
    _, _, jy2, _ = angular(2)
    for beta in np.linspace(0, pi, 101):
        c, s = np.cos(beta), np.sin(beta)
        analytic = np.array([np.sqrt(3 / 8) * s * s, -np.sqrt(3 / 2) * s * c,
                             (3 * c * c - 1) / 2, np.sqrt(3 / 2) * s * c,
                             np.sqrt(3 / 8) * s * s])
        harmonic = np.sqrt(4 * pi / 5) * np.array([
            ylm(2, m, beta, 0.0).conjugate() for m in (2, 1, 0, -1, -2)
        ])
        record("exercise_3_26_revisit_three_routes",
               [np.max(abs(analytic - harmonic)),
                np.max(abs(analytic - rotation(jy2, beta)[:, 2]))])
        record("exercise_3_26_revisit_total_probability", np.sum(abs(analytic) ** 2) - 1)

    return {
        "exercise_3_23_m_order": [1, 0, -1],
        "exercise_3_23_probabilities": (np.array([1, 9, 1]) / 11).tolist(),
        "sphere_quadrature": "80-point Gauss-Legendre in u=cos(theta), 128 uniform phi points",
        "matrix_units": "hbar=1",
    }


def tidy(ax):
    ax.grid(True, color="#e3e9f0", linewidth=0.8)
    ax.set_axisbelow(True)


def theta_ticks(ax):
    ax.set_xlim(0, pi)
    ax.set_xticks([0, pi / 4, pi / 2, 3 * pi / 4, pi],
                  ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])


def save_figure(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{name}.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def probability_measure_figure():
    theta = np.linspace(0, pi, 801)
    phi = np.linspace(0, 2 * pi, 801)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.6))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.23, top=0.73, wspace=0.30)
    fig.suptitle("球谐函数的概率：先说明对什么测度求密度", fontsize=17, y=0.965)
    fig.text(0.5, 0.855,
             r"$dP=|Y_l^m(\theta,\phi)|^2\,d\Omega"
             r"=|Y_l^m(\theta,\phi)|^2\sin\theta\,d\theta\,d\phi$",
             ha="center", fontsize=16)
    choices = [(0, 0, BLUE), (1, 0, ORANGE), (1, 1, GREEN)]
    for l, m, color in choices:
        density = abs(ylm(l, m, theta, 0.0)) ** 2
        label = rf"$Y_{{{l}}}^{{{m}}}$"
        axes[0].plot(theta, density, color=color, lw=2.5, label=label)
        axes[1].plot(theta, 2 * pi * np.sin(theta) * density, color=color, lw=2.5, label=label)
    axes[0].set_title("① 每单位立体角的密度", pad=14)
    axes[0].set_ylabel(r"$|Y_l^m|^2$（sr$^{-1}$）")
    axes[0].set_xlabel(r"极角 $\theta$")
    axes[0].set_ylim(0, 0.265)
    axes[0].legend(frameon=False, loc="upper center", ncol=1)
    axes[1].set_title("② 只记录极角时的密度", pad=14)
    axes[1].set_ylabel(r"$p_\theta=2\pi\sin\theta\,|Y_l^m|^2$")
    axes[1].set_xlabel(r"极角 $\theta$")
    axes[1].set_ylim(0, 0.84)
    axes[1].legend(frameon=False, loc="upper left")
    for ax in axes[:2]:
        theta_ticks(ax)
        tidy(ax)
    axes[2].plot(phi, np.full_like(phi, 1 / (2 * pi)), color=GREEN, lw=2.5,
                 label=r"复球谐 $Y_1^1$")
    axes[2].plot(phi, np.cos(phi) ** 2 / pi, color=PURPLE, lw=2.5,
                 label=r"实轨道 $p_x$")
    axes[2].set_title("③ 只记录方位角时的密度", pad=14)
    axes[2].set_ylabel(r"$p_\phi=\int_0^\pi |Y|^2\sin\theta\,d\theta$")
    axes[2].set_xlabel(r"方位角 $\phi$")
    axes[2].set_xlim(0, 2 * pi)
    axes[2].set_xticks([0, pi / 2, pi, 3 * pi / 2, 2 * pi],
                      ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axes[2].set_ylim(0, 0.44)
    axes[2].legend(frameon=False, loc="upper center")
    tidy(axes[2])
    fig.text(0.065, 0.105, "① 的三态密度均与 φ 无关；② 已对 φ 积分，曲线下面积各为 1。", fontsize=11)
    fig.text(0.065, 0.055,
             r"③ 中 $p_x=(Y_1^{-1}-Y_1^1)/\sqrt{2}$；实轨道的方位变化来自两种 $m$ 振幅的相干叠加。",
             fontsize=11)
    save_figure(fig, "3.6-球谐函数与角概率")


def real_p_surface(ax, beta, title):
    theta = np.linspace(0, pi, 101)
    phi = np.linspace(0, 2 * pi, 181)
    theta, phi = np.meshgrid(theta, phi, indexing="ij")
    nx, ny, nz = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    projection = np.sin(beta) * nx + np.cos(beta) * nz
    radial = projection ** 2  # |ψ|²/max|ψ|²，清楚注明使用共同缩放。
    positive = np.array(matplotlib.colors.to_rgba(BLUE))
    negative = np.array(matplotlib.colors.to_rgba(ORANGE))
    colors = np.where((projection >= 0)[..., None], positive, negative)
    ax.plot_surface(radial * nx, radial * ny, radial * nz,
                    facecolors=colors, rstride=2, cstride=2,
                    linewidth=0, antialiased=True, shade=True)
    for direction, label in [(np.array([1, 0, 0]), "x"),
                             (np.array([0, 1, 0]), "y"),
                             (np.array([0, 0, 1]), "z")]:
        a, b = -1.1 * direction, 1.17 * direction
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#8c98a5", lw=0.75)
        same_as_axis = ((label == "z" and np.isclose(beta, 0.0))
                        or (label == "x" and np.isclose(beta, pi / 2)))
        if not same_as_axis:
            ax.text(*(1.27 * direction), label, color="#5c6875", fontsize=10)
    axis = np.array([np.sin(beta), 0.0, np.cos(beta)])
    ax.quiver(0, 0, 0, *(1.2 * axis), color="#182c43",
              linewidth=1.5, arrow_length_ratio=0.12)
    axis_label = "z = n" if np.isclose(beta, 0.0) else ("x = n" if np.isclose(beta, pi / 2) else "n")
    ax.text(*(1.39 * axis), axis_label, color="#182c43", fontsize=11)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-57)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, y=0.98)


def rotation_figure():
    fig = plt.figure(figsize=(13.6, 9.7))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25],
                          left=0.08, right=0.96, bottom=0.14, top=0.81,
                          hspace=0.30, wspace=0.0)
    fig.suptitle("旋转实 p 轨道：图案方向与角动量测量", fontsize=17, y=0.96)
    fig.text(0.5, 0.89,
             r"$U_y(\beta)|1,0\rangle$：仍然是 $l=1$，测量固定 $z$ 轴的 $L_z$ 时各结果概率随 $\beta$ 改变",
             ha="center", fontsize=12)
    ax = fig.add_subplot(grid[0, :])
    beta = np.linspace(0, pi, 801)
    ax.plot(beta, np.cos(beta) ** 2, color=BLUE, lw=2.8, label=r"$m=0:\ \cos^2\beta$")
    ax.plot(beta, np.sin(beta) ** 2 / 2, color=ORANGE, lw=2.8,
            label=r"$m=+1$ 与 $m=-1$（各自）$: \sin^2\beta/2$")
    for angle in [0, pi / 4, pi / 2]:
        ax.axvline(angle, color="#9aa8b7", lw=0.8, ls="--", zorder=0)
    theta_ticks(ax)
    ax.set_xlabel(r"绕 $y$ 轴的旋转角 $\beta$")
    ax.set_ylabel(r"测量 $L_z=m\hbar$ 的概率")
    ax.set_ylim(-0.02, 1.07)
    ax.legend(loc="upper center", frameon=False)
    tidy(ax)
    for i, (angle, label) in enumerate([
        (0.0, r"$\beta=0$：实 $p_z$"),
        (pi / 4, r"$\beta=\pi/4$：倾斜的实 $p$"),
        (pi / 2, r"$\beta=\pi/2$：实 $p_x$"),
    ]):
        real_p_surface(fig.add_subplot(grid[1, i], projection="3d"), angle, label)
    fig.text(0.08, 0.092,
             r"下排曲面半径为 $|\psi(\hat r)|^2/\max|\psi|^2$，三个面采用同一尺度；蓝 / 橙仅表示波函数正 / 负号。",
             fontsize=11)
    fig.text(0.08, 0.043,
             r"箭头 n 是图案轴，不是非零平均角动量：$L_n\psi=0,\quad\langle\mathbf{L}\rangle=0,\quad L^2\psi=2\hbar^2\psi$。",
             fontsize=12)
    save_figure(fig, "3.6-轨道旋转与球谐叠加")


def main():
    facts = numerical_checks()
    probability_measure_figure()
    rotation_figure()
    report = {
        "section": "3.6 轨道角动量",
        "purpose": "教学补充及解析公式复算；不是本节原题规定的数值任务",
        "phase_convention": "Condon–Shortley, active rotation exp(-i beta Ly/hbar)",
        "probability_measure": "dP=|Y|^2 dOmega=|Y|^2 sin(theta) dtheta dphi",
        "status": "passed",
        "all_checks_passed": all(item["passed"] for item in CHECKS.values()),
        "checks": CHECKS,
        "reference_values": facts,
        "outputs": [
            "3.6-球谐函数与角概率.png", "3.6-球谐函数与角概率.svg",
            "3.6-轨道旋转与球谐叠加.png", "3.6-轨道旋转与球谐叠加.svg",
        ],
    }
    (OUT / "3.6-球谐函数与轨道旋转核验结果.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(CHECKS),
                      "outputs": report["outputs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
