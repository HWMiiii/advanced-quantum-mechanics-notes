"""Sakurai 中文第3版 §3.7，原题3.28：球形势阱解析式与指定数值。

原题必做：(a)验证l=0,1,2无限球阱能量；(b)β=4,10,25,100的l=0最低能量与无限壁极限。
教学补充：阈值、波函数归一、阱外概率及两张图。
无量纲约定：Ea=ħ²/(2Ma²)，x=ka，y=κa，V0/Ea=β²，E/Ea=x²；
井内V=0，井外V=V0。结果保存在本脚本所在目录。
依赖NumPy、Matplotlib；无需SciPy。解析模型不是实验数据，不代表学习者已独立作答。
"""
from pathlib import Path
from math import pi, sin, cos, sqrt, asin
import csv
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
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.unicode_minus": False, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
})
COLORS = ["#2867a8", "#d07835", "#238577", "#8660a3", "#57616e"]
CHECKS = {}


def record(name, error, tolerance=2e-10):
    error = float(np.max(np.abs(error)))
    error = max(error, CHECKS.get(name, {}).get("max_abs_error", 0.0))
    CHECKS[name] = {"max_abs_error": error, "tolerance": tolerance,
                    "passed": bool(error <= tolerance)}
    if error > tolerance:
        raise AssertionError((name, error, tolerance))


def bisect(function, low, high, xtol=2e-14, maxiter=200):
    fl, fh = function(low), function(high)
    if fl == 0:
        return low
    if fh == 0:
        return high
    if fl * fh >= 0:
        raise ValueError(("No sign change", low, high, fl, fh))
    for _ in range(maxiter):
        middle = (low + high) / 2
        fm = function(middle)
        if fm == 0 or high - low < xtol:
            return middle
        if fl * fm < 0:
            high = middle
        else:
            low, fl = middle, fm
    raise RuntimeError("Bisection did not converge")


def j_spherical(l, x):
    x = np.asarray(x, dtype=float)
    if l == 0:
        return np.sin(x) / x
    if l == 1:
        return np.sin(x) / x**2 - np.cos(x) / x
    if l == 2:
        return (3 / x**3 - 1 / x) * np.sin(x) - 3 * np.cos(x) / x**2
    raise ValueError(l)


def infinite_roots():
    rows = []
    quoted = {0: [pi, 2*pi, 3*pi], 1: [4.49, 7.73, 10.90], 2: [5.76, 9.10, 12.32]}
    for l in [0, 1, 2]:
        for n in [1, 2, 3]:
            if l == 0:
                root = n * pi
            elif l == 1:
                root = bisect(lambda z: sin(z) - z*cos(z), n*pi, (n+0.5)*pi)
            else:
                root = bisect(lambda z: (3-z*z)*sin(z) - 3*z*cos(z),
                              (n+0.5)*pi, (n+1)*pi)
            residual = float(j_spherical(l, root))
            record("infinite_spherical_bessel_roots", residual, 3e-13)
            if l > 0:
                record("textbook_two_decimal_root_values", root - quoted[l][n-1], 0.0051)
            rows.append({
                "l": l, "radial_root_number": n, "ka": root,
                "E_over_Ea": root*root, "magnetic_degeneracy": 2*l+1,
                "j_l_residual": residual, "textbook_ka_display": quoted[l][n-1],
            })
    return rows


def finite_state(beta):
    if beta <= pi/2:
        raise ValueError("For an l=0 bound state beta must exceed pi/2")
    high = min(pi-1e-10, beta-1e-10)
    root_function = lambda x: x / np.tan(x) + np.sqrt(beta*beta-x*x)
    x = bisect(root_function, pi/2 + 1e-10, high)
    y = sqrt(beta*beta-x*x)
    # Independent root equation: x+arcsin(x/beta)=pi in the ground-state branch.
    x_phase = bisect(lambda z: z+asin(z/beta)-pi, pi/2+1e-10, high)
    norm_inside = 0.5 - sin(2*x)/(4*x)
    norm_outside = sin(x)**2/(2*y)
    norm_squared = 1/(norm_inside+norm_outside)
    record("finite_matching_equation", root_function(x), 3e-10)
    record("finite_independent_phase_equation", x + asin(x/beta) - pi, 2e-13)
    record("finite_two_root_methods", x - x_phase, 5e-13)
    record("finite_wavefunction_derivative_matching", x*cos(x)+y*sin(x), 3e-11)
    record("finite_norm_analytic", norm_squared*(norm_inside+norm_outside)-1, 3e-15)
    return {
        "beta": float(beta), "V0_over_Ea": float(beta*beta),
        "ka": x, "kappa_a": y, "E_over_Ea": x*x,
        "E_over_V0": x*x/(beta*beta),
        "E_over_E_infinite": x*x/(pi*pi),
        "relative_deficit_to_infinite": 1-x*x/(pi*pi),
        "outside_probability_teaching_supplement": norm_squared*norm_outside,
        "dimensionless_radial_amplitude": sqrt(norm_squared),
        "matching_residual": float(root_function(x)),
        "independent_phase_residual": x+asin(x/beta)-pi,
    }


def radial_wave(t, state):
    t = np.asarray(t)
    x, y = state["ka"], state["kappa_a"]
    normalization = state["dimensionless_radial_amplitude"]
    result = normalization * np.sin(x * np.minimum(t, 1))
    outside = t > 1
    result[outside] = normalization*sin(x)*np.exp(-y*(t[outside]-1))
    return result


def save_csv(name, rows):
    with (OUT/name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, name):
    fig.savefig(OUT/f"{name}.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(OUT/f"{name}.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def style_axis(ax):
    ax.grid(True, color="#e3e9f0", lw=0.8)
    ax.set_axisbelow(True)


def root_and_convergence(states):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8))
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.23, top=0.77, wspace=0.25)
    fig.suptitle("原题 3.28：有限球形势阱的基态根与无限壁极限", fontsize=17, y=0.96)
    fig.text(0.5, 0.875,
             r"$V(r<a)=0,\quad V(r>a)=V_0=\beta^2E_a,\quad E_a=\hbar^2/(2Ma^2),\quad E/E_a=x^2$",
             ha="center", fontsize=12)
    state = states[0]
    xx = np.linspace(pi/2+0.001, pi-0.001, 1000)
    axes[0].plot(xx, -xx/np.tan(xx), color=COLORS[0], lw=2.4,
                 label=r"$-x\cot x$")
    axes[0].plot(xx, np.sqrt(16-xx*xx), color=COLORS[1], lw=2.4,
                 label=r"$\sqrt{\beta^2-x^2},\ \beta=4$")
    axes[0].scatter([state["ka"]], [state["kappa_a"]], color="#152c47", s=48, zorder=4)
    axes[0].annotate(rf"$x={state['ka']:.6f}$",
                     (state["ka"], state["kappa_a"]), xytext=(-100, -40),
                     textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "#536779"},
                     fontsize=11)
    axes[0].set_xlim(pi/2, pi)
    axes[0].set_ylim(0, 8)
    axes[0].set_xticks([pi/2, 2, 2.5, 3, pi],
                       [r"$\pi/2$", "2", "2.5", "3", r"$\pi$"])
    axes[0].set_xlabel(r"$x=ka$")
    axes[0].set_ylabel("无量纲匹配值")
    axes[0].set_title("① 根在最低支内唯一", pad=14)
    axes[0].legend(frameon=False, loc="upper left")
    betas = np.geomspace(4, 150, 260)
    curve = np.array([finite_state(b)["E_over_E_infinite"] for b in betas])
    axes[1].semilogx(betas, curve, color=COLORS[0], lw=2.5)
    axes[1].axhline(1, color="#57616e", lw=1.2, ls="--", label=r"无限壁：$E/E_a=\pi^2$")
    axes[1].scatter([s["beta"] for s in states], [s["E_over_E_infinite"] for s in states],
                    color=COLORS[1], s=42, zorder=4, label="原题指定的四个参数")
    axes[1].set_xlim(3.5, 160)
    axes[1].set_ylim(0.55, 1.045)
    axes[1].set_xticks([4, 10, 25, 100], ["4", "10", "25", "100"])
    axes[1].set_xlabel(r"阱壁参数 $\beta$（对数横轴）")
    axes[1].set_ylabel(r"$E/E_\infty=x^2/\pi^2$")
    axes[1].set_title("② 能量由下方趋近无限壁基态", pad=14)
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        style_axis(ax)
    fig.text(0.075, 0.105, "原题要求四组数值；连续曲线和左侧根图是教学补充。有限势阱允许波函数进入阱外。", fontsize=11)
    fig.text(0.075, 0.053,
             r"根方程：$x\cot x=-\sqrt{\beta^2-x^2}$；最低束缚态要求 $\beta>\pi/2$，四组题设均满足。",
             fontsize=11)
    save_figure(fig, "3.7-球阱根方程与收敛")


def levels_and_probability(infinite, states):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.1))
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.25, top=0.78, wspace=0.25)
    fig.suptitle("球阱的角动量能谱与有限壁基态的径向概率", fontsize=17, y=0.965)
    fig.text(0.5, 0.885,
             r"左：无限壁的 $l=0,1,2$ 前三个径向能级；右：$\int_0^\infty |w(t)|^2dt=1,\ t=r/a,\ w=\sqrt{a}\,u$",
             ha="center", fontsize=12)
    for row in infinite:
        l, n, energy = row["l"], row["radial_root_number"], row["E_over_Ea"]
        axes[0].hlines(energy, l-0.27, l+0.27, color=COLORS[n-1], lw=2.5)
        axes[0].text(l+0.31, energy, f"{energy:.2f}", va="center", fontsize=9.5)
    axes[0].set_xlim(-0.48, 2.7)
    axes[0].set_ylim(0, 166)
    axes[0].set_xticks([0, 1, 2], [r"$l=0$", r"$l=1$", r"$l=2$"])
    axes[0].set_ylabel(r"无限壁能量 $E/E_a$")
    axes[0].set_xlabel(r"每条能级的磁量子数简并度为 $2l+1$")
    axes[0].set_title("① 不同 l 的零点决定不同能量", pad=14)
    axes[0].grid(axis="y", color="#e3e9f0", lw=0.8)
    t = np.linspace(0, 2.0, 1601)
    axes[1].axvspan(1, 2, color="#eaf0f7", zorder=0)
    for state, color in zip(states, COLORS):
        axes[1].plot(t, radial_wave(t, state)**2, color=color, lw=2.2,
                     label=rf"$\beta={int(state['beta'])}$")
    infinite_density = np.where(t <= 1, 2*np.sin(pi*t)**2, 0)
    axes[1].plot(t, infinite_density, color=COLORS[4], lw=1.8, ls="--", label="无限壁")
    axes[1].axvline(1, color="#8895a5", lw=1.0, ls=":")
    axes[1].text(1.52, 2.02, "阱外", ha="center", color="#526477")
    axes[1].set_xlim(0, 2)
    axes[1].set_ylim(0, 2.18)
    axes[1].set_xlabel(r"$t=r/a$")
    axes[1].set_ylabel(r"每单位 $t$ 的径向概率密度 $|w(t)|^2$")
    axes[1].set_title("② 阱壁有限时，概率可延伸到阱外", pad=14)
    axes[1].legend(frameon=False, loc="upper right", ncol=1)
    style_axis(axes[1])
    fig.text(0.075, 0.115,
             r"右图是已对角度积分后的径向密度：$|u(r)|^2dr=|w(t)|^2dt$，不等于三维位置概率密度 $|\psi(\mathbf{x})|^2$。",
             fontsize=11)
    fig.text(0.075, 0.060, "原题(a)的前三个零点已独立求根；右图及阱外概率用于解释有限壁为何降低基态能量。", fontsize=11)
    save_figure(fig, "3.7-球阱能谱与径向概率")


def main():
    infinite = infinite_roots()
    finite = [finite_state(beta) for beta in [4, 10, 25, 100]]
    # Simpson替代并非必需：高分辨率梯形积分与解析归一独立对照。
    for state in finite:
        t = np.linspace(0, 1+20/state["kappa_a"], 100001)
        record("radial_probability_quadrature", np.trapezoid(radial_wave(t, state)**2, t)-1, 2e-9)
    save_csv("3.7-无限球阱零点数值.csv", infinite)
    save_csv("3.7-有限球阱基态数值.csv", finite)
    root_and_convergence(finite)
    levels_and_probability(infinite, finite)
    report = {
        "section": "3.7 中心势的薛定谔方程",
        "source": "习题3.28 PDF261/印刷239；正文式3.287-3.289 PDF217/印刷195",
        "required_parameters": [4, 10, 25, 100],
        "energy_zero": "V_inside=0, V_outside=V0; Ea=hbar^2/(2 M a^2)",
        "method": "bracketed bisection; independent phase equation checks the finite ground-state branch",
        "original_exercise_required": "l=0,1,2 infinite well spectra and four finite l=0 ground energies",
        "teaching_supplements": "continuous convergence curve, roots plot, radial density and leakage probability",
        "status": "passed",
        "all_checks_passed": all(c["passed"] for c in CHECKS.values()),
        "checks": CHECKS,
        "infinite_well": infinite,
        "finite_well": finite,
    }
    (OUT/"3.7-球形势阱数值核验结果.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "finite_well": finite,
                      "infinite_well": infinite}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
