"""§3.4 教学配图与独立数值核验；所有曲线均由本节公式计算，无实验数据。

依赖 Python 3、NumPy、Matplotlib。运行后在本文件所在目录生成 PNG、SVG 和 JSON。
数值检验用于交叉检查，不能替代笔记中的解析证明。
"""
from pathlib import Path
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
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.unicode_minus": False, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#f8fafc", "axes.facecolor": "#ffffff",
})
BLUE, ORANGE, GREEN = "#2867a8", "#cc6b31", "#238577"
I = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1, -1]).astype(complex)
PAULI = [SX, SY, SZ]
RNG = np.random.default_rng(20261023)
checks = {}

def record(name, error, tolerance=1e-11):
    error = float(abs(error))
    checks[name] = {"max_abs_error": error, "tolerance": tolerance, "passed": error <= tolerance}
    assert error <= tolerance, (name, error, tolerance)

def entropy(rho):
    p = np.linalg.eigvalsh(rho)
    p = p[p > 1e-14]
    return float(-np.dot(p, np.log(p)))

def projector(psi):
    return np.outer(psi, psi.conj())

def hermitian(n):
    z = RNG.normal(size=(n, n)) + 1j * RNG.normal(size=(n, n))
    return (z + z.conj().T) / 2

def unitary(h, t):
    e, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * e * t)) @ v.conj().T

def random_density(n):
    z = RNG.normal(size=(n, n)) + 1j * RNG.normal(size=(n, n))
    rho = z @ z.conj().T
    return rho / np.trace(rho)

def quantum_checks():
    maxima = dict.fromkeys(["trace_formula", "normalization", "hermiticity", "purity_overlap",
                            "positivity_violation", "purity_bound_violation",
                            "unitary_spectrum", "unitary_entropy", "unitary_purity",
                            "von_neumann_sign"], 0.0)
    for _ in range(100):
        n = int(RNG.integers(2, 6))
        k = n + 3
        psi = RNG.normal(size=(n, k)) + 1j * RNG.normal(size=(n, k))
        psi /= np.linalg.norm(psi, axis=0)
        w = RNG.random(k)
        w /= w.sum()
        rho = (psi * w) @ psi.conj().T
        a = hermitian(n)
        avg = sum(w[j] * (psi[:, j].conj() @ a @ psi[:, j]) for j in range(k))
        overlap = psi.conj().T @ psi
        purity = np.trace(rho @ rho).real
        h, t = hermitian(n), 0.37
        u = unitary(h, t)
        rt = u @ rho @ u.conj().T
        dt = 2e-5
        up, um = unitary(h, t + dt), unitary(h, t - dt)
        derivative = (up @ rho @ up.conj().T - um @ rho @ um.conj().T) / (2 * dt)
        vals = {
            "trace_formula": abs(np.trace(rho @ a) - avg),
            "normalization": abs(np.trace(rho) - 1),
            "hermiticity": np.max(abs(rho - rho.conj().T)),
            "purity_overlap": abs(purity - np.sum(w[:, None] * w * abs(overlap)**2)),
            "positivity_violation": max(0.0, -np.linalg.eigvalsh(rho).min()),
            "purity_bound_violation": max(0.0, 1/n - purity, purity - 1),
            "unitary_spectrum": np.max(abs(np.linalg.eigvalsh(rho) - np.linalg.eigvalsh(rt))),
            "unitary_entropy": abs(entropy(rho) - entropy(rt)),
            "unitary_purity": abs(purity - np.trace(rt @ rt)),
            "von_neumann_sign": np.max(abs(1j * derivative - (h @ rt - rt @ h))),
        }
        for key, value in vals.items():
            maxima[key] = max(maxima[key], float(value))
    for key, value in maxima.items():
        record(key, value, 5e-8 if key == "von_neumann_sign" else 1e-11)

    plusz = np.array([1, 0], complex)
    plusx = np.array([1, 1], complex) / np.sqrt(2)
    rho = 0.75 * projector(plusz) + 0.25 * projector(plusx)
    r = np.array([0.25, 0, 0.75])
    axis = np.array([1, 0, 3]) / np.sqrt(10)
    n_sigma = sum(axis[j] * PAULI[j] for j in range(3))
    pplus, pminus = (4 + np.sqrt(10))/8, (4 - np.sqrt(10))/8
    rho_spectral = pplus * (I+n_sigma)/2 + pminus * (I-n_sigma)/2
    record("textbook_75_25_matrix", np.max(abs(rho - np.array([[7, 1], [1, 1]])/8)))
    record("textbook_75_25_spectral_decomposition", np.max(abs(rho - rho_spectral)))
    record("textbook_75_25_purity", np.trace(rho @ rho).real - 13/16)
    record("textbook_75_25_bloch", np.max(abs([np.trace(rho @ s).real for s in PAULI] - r)))
    record("nonorthogonal_overlap_formula", np.trace(rho @ rho).real -
           (0.75**2 + 0.25**2 + 2*0.75*0.25*abs(plusz.conj() @ plusx)**2))

    max_bloch, max_prob, max_gibbs_entropy, min_rel = 0., 0., 0., np.inf
    for _ in range(100):
        rho = random_density(2)
        r = np.array([np.trace(rho @ s).real for s in PAULI])
        reconstructed = (I + sum(r[j] * PAULI[j] for j in range(3))) / 2
        max_bloch = max(max_bloch, np.max(abs(rho-reconstructed)),
                        abs(np.trace(rho@rho).real-(1+r@r)/2))
        axis = RNG.normal(size=3)
        axis /= np.linalg.norm(axis)
        proj = (I+sum(axis[j]*PAULI[j] for j in range(3)))/2
        max_prob = max(max_prob, abs(np.trace(rho@proj)-(1+r@axis)/2))
        h = hermitian(3)
        energies, v = np.linalg.eigh(h)
        beta = 0.8
        z = np.exp(-beta*energies).sum()
        probs = np.exp(-beta*energies)/z
        tau = (v*probs) @ v.conj().T
        avg = np.trace(tau@h).real
        max_gibbs_entropy = max(max_gibbs_entropy, abs(entropy(tau)-beta*avg-np.log(z)))
        candidate = random_density(3)
        relative = -entropy(candidate)+beta*np.trace(candidate@h).real+np.log(z)
        min_rel = min(min_rel, relative)
    record("bloch_reconstruction_and_purity", max_bloch)
    record("bloch_projector_probability", max_prob)
    record("gibbs_entropy_identity", max_gibbs_entropy)
    record("relative_entropy_negativity", max(0., -min_rel))

    phases = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    time_average = np.array([[0.5, 0.5*np.exp(-1j*phases).mean()],
                             [0.5*np.exp(1j*phases).mean(), 0.5]])
    record("full_period_time_average", np.max(abs(time_average-I/2)))

    # ℏ=1。连续高斯核可解析求导；先对x求导，再取x'=x。
    s, k = 0.8, 1.1
    x = np.linspace(-10*s, 10*s, 40001)
    phi = (2*np.pi*s*s)**(-0.25)*np.exp(-x*x/(4*s*s))
    psi_plus, psi_minus = phi*np.exp(1j*k*x), phi*np.exp(-1j*k*x)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    record("gaussian_normalization", trap(phi**2, x)-1)
    overlap = trap(psi_minus.conj()*psi_plus, x)
    record("gaussian_overlap", overlap-np.exp(-2*k*k*s*s))
    dphi = -x*phi/(2*s*s)
    pure_p = -1j*trap((dphi+1j*k*phi)*phi, x)
    mixture_p = -1j*trap(dphi*phi, x)
    record("gaussian_pure_momentum", pure_p-k)
    record("gaussian_mixed_momentum", mixture_p)
    record("gaussian_mixture_purity", (1+abs(overlap)**2)/2-(1+np.exp(-4*k*k*s*s))/2)

    x = np.linspace(0, 4, 301)
    thermal_up = 1/(np.exp(2*x)+1)
    thermal_down = 1-thermal_up
    record("thermal_bloch_polarization", np.max(abs(thermal_up-thermal_down+np.tanh(x))))
    ent = -(thermal_up*np.log(thermal_up)+thermal_down*np.log(thermal_down))
    record("thermal_spin_entropy", np.max(abs(ent-np.log(2*np.cosh(x))+x*np.tanh(x))))
    delta = 1e-5
    deriv = (np.tanh(x+delta)-np.tanh(x-delta))/(2*delta)
    record("finite_field_differential_response", np.max(abs(deriv-1/np.cosh(x)**2)), 1e-9)
    small = 1e-3
    oscillator_energy = 0.5 + 1/np.expm1(small)
    record("oscillator_high_temperature_beta_U", small*oscillator_energy-1, 1e-7)

    return {
        "random_ensemble_samples": 100,
        "random_bloch_and_gibbs_samples": 100,
        "textbook_75_25_spectral_weights": [float(pplus), float(pminus)],
        "textbook_75_25_purity": 13/16,
        "textbook_75_25_von_neumann_entropy": entropy(np.array([[7, 1], [1, 1]])/8),
        "gibbs_relative_entropy_minimum_in_samples": float(min_rel),
        "gaussian_example": {"s": s, "k": k, "hbar": 1,
                             "pure_momentum": float(pure_p.real),
                             "mixed_momentum": float(mixture_p.real)},
    }

def style(ax):
    ax.grid(alpha=0.17)
    ax.set_axisbelow(True)

def save(fig, name):
    fig.savefig(OUT/(name+".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT/(name+".svg"), bbox_inches="tight")
    plt.close(fig)

def figures():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    theta = np.linspace(0, 2*np.pi, 600)
    ax = axes[0]
    ax.plot(theta, (1+np.sin(theta))/2, color=BLUE, lw=2.5, label=r"$|+x\rangle$ 纯态")
    ax.plot(theta, np.full_like(theta, 0.5), color=ORANGE, lw=2.3, ls="--", label=r"$I_2/2$ 未极化混合态")
    ax.set(title="转动分析器，可以区分两种制备",
           xlabel=r"分析器方向 $\mathbf{n}=(\sin\theta,0,\cos\theta)$", ylabel=r"$P(+\mathbf{n})$")
    ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],
                  [r"$0$"+ "\n+z",r"$\pi/2$"+"\n+x",r"$\pi$"+"\n−z",r"$3\pi/2$"+"\n−x",r"$2\pi$"+"\n+z"])
    ax.set_ylim(-0.04, 1.07)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    style(ax)
    r = np.linspace(0, 1, 601)
    p, q = (1+r)/2, (1-r)/2
    ent = -p*np.log(p)
    mask = q > 0
    ent[mask] -= q[mask]*np.log(q[mask])
    ax = axes[1]
    ax.plot(r, (1+r*r)/2, color=BLUE, lw=2.5, label=r"纯度 $\mathrm{Tr}\rho^2$")
    ax.plot(r, ent/np.log(2), color=GREEN, lw=2.5, label=r"归一化熵 $\sigma_{\rm ent}/\ln2$")
    ax.set(title="Bloch 向量长度刻画混合程度", xlabel=r"Bloch 向量长度 $|\mathbf{r}|$", ylabel="无量纲量")
    ax.set_ylim(-0.04, 1.07)
    ax.legend(loc="center left", fontsize=10)
    style(ax)
    fig.suptitle("§3.4  相干叠加、混合系综与纯度", fontsize=16)
    save(fig, "3.4-相干叠加与混合系综")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    x = np.linspace(0, 4, 601)
    up = 1/(np.exp(2*x)+1)
    ax = axes[0]
    ax.plot(x, up, color=ORANGE, lw=2.5, label=r"$S_z=+\hbar/2,\ E=+\mu_BB$")
    ax.plot(x, 1-up, color=BLUE, lw=2.5, label=r"$S_z=-\hbar/2,\ E=-\mu_BB$")
    ax.set(title="电子低能态：自旋反向，磁矩顺场",
           xlabel=r"$x=\beta\mu_B B$ （固定 $B>0$ 时，向右温度降低）", ylabel="热平衡占据概率")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center right", fontsize=10)
    style(ax)
    ax = axes[1]
    ratio = np.divide(np.tanh(x), x, out=np.ones_like(x), where=x!=0)
    ax.plot(x, np.tanh(x), color=BLUE, lw=2.5, label=r"$m_z/\mu_B=\tanh x$")
    ax.plot(x, ratio, color=ORANGE, lw=2.3, ls="--", label=r"比值响应：$\tanh x/x$")
    ax.plot(x, 1/np.cosh(x)**2, color=GREEN, lw=2.5, ls="-.", label=r"微分响应：$\mathrm{sech}^2x$")
    ax.set(title=r"两种响应均除以 $\beta\mu_B^2$", xlabel=r"$x=\beta\mu_B B$", ylabel="归一化磁矩 / 归一化响应")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center right", fontsize=10)
    style(ax)
    fig.suptitle("§3.4  热平衡自旋与有限场磁响应", fontsize=16)
    save(fig, "3.4-热平衡自旋与磁响应")

if __name__ == "__main__":
    examples = quantum_checks()
    figures()
    result = {"status": "PASS", "section": "3.4", "seed": 20261023,
              "meaning": "由教材与补充页公式独立复算；不代表实验数据，也不代表学习者已掌握。",
              "checks": checks, "examples": examples}
    target = OUT/"3.4-密度算符与统计核验结果.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "figures": 2, "result": str(target)}, ensure_ascii=False))
