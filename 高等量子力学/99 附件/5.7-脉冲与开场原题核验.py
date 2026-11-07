"""数值复核5.30、5.31、5.34、5.38。仅依赖NumPy，结果写在脚本同目录。"""

import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


def evolve_oscillator(matrix, tau, end_time, steps):
    """直接积分14态截断的含时方程；m=ℏ=ω=1，省去自由零点能。"""
    levels = np.arange(len(matrix))
    amplitudes = np.zeros(len(levels), dtype=complex)
    amplitudes[0] = 1
    dt = end_time / steps

    def derivative(time, state):
        phases = np.exp(1j * levels * time)
        return -1j * np.exp(-time / tau) * phases * (matrix @ (phases.conj() * state))

    for index in range(steps):
        time = index * dt
        k1 = derivative(time, amplitudes)
        k2 = derivative(time + dt / 2, amplitudes + dt * k1 / 2)
        k3 = derivative(time + dt / 2, amplitudes + dt * k2 / 2)
        k4 = derivative(time + dt, amplitudes + dt * k3)
        amplitudes += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return np.abs(amplitudes) ** 2


def gauss_integral(function, start, end, nodes, weights):
    times = start + (end - start) * (nodes + 1) / 2
    return (end - start) * np.sum(weights * function(times)) / 2


def oscillatory_integral(freq, end):
    # J(k,t) = ∫_0^t exp(ik s) ds；用sinc稳定处理零频和近零频。
    return end * np.exp(0.5j * freq * end) * np.sinc(freq * end / (2 * np.pi))


def complex_pair(value):
    return [float(np.real(value)), float(np.imag(value))]


def main():
    report = {"units": "m = ℏ = ω = 1", "5.30": [], "5.31": [], "5.38": []}
    checks = []

    def check(label, value, tolerance):
        checks.append({"name": label, "max_abs_error": float(value),
                       "tolerance": tolerance, "passed": bool(value <= tolerance)})

    n_levels = 14
    lowering = np.diag(np.sqrt(np.arange(1, n_levels)), 1)
    x = (lowering + lowering.T) / math.sqrt(2)
    tau, force = 0.8, 0.12
    for time in (1.6, 16.0):
        actual = evolve_oscillator(-force * x, tau, time, 1800)
        alpha = (1j * force / math.sqrt(2) *
                 (1 - np.exp((-1 / tau + 1j) * time)) / (1 / tau - 1j))
        b = abs(alpha) ** 2
        exact = [math.exp(-b) * b**n / math.factorial(n) for n in range(n_levels)]
        error = np.max(np.abs(actual - exact))
        report["5.30"].append({"t": time, "tau": tau, "F0": force, "P1_leading": b,
                              "P_n_ode": actual.tolist(), "P_n_coherent": exact})
        check(f"5.30 t={time}，完整分布与14态直接演化", error, 1e-11)

    ratios = []
    for amplitude in (0.02, 0.01, 0.005):
        actual = evolve_oscillator(amplitude * (x @ x), tau, 16.0, 2400)
        sigma = (amplitude / 2) ** 2 / (tau**-2 + 4)
        ratios.append([actual[2] / (2 * sigma), actual[4] / (6 * sigma**2)])
        report["5.31"].append({"A": amplitude, "tau": tau, "t": 16.0,
                              "P_n_ode": actual.tolist(), "P2_leading": 2 * sigma,
                              "P4_leading": 6 * sigma**2, "ode_over_leading": ratios[-1]})
        check(f"5.31 A={amplitude}，奇态禁戒", np.max(actual[1::2]), 1e-14)
        check(f"5.31 A={amplitude}，概率和", abs(sum(actual) - 1), 1e-10)
    for level, errors in zip((2, 4), np.abs(np.array(ratios) - 1).T):
        check(f"5.31 P{level}，弱场首项相对差依次缩小", max(0.0, max(np.diff(errors))), 0)
        check(f"5.31 P{level}，最弱场首项相对差", errors[-1], 0.01)

    nodes, weights = leggauss(160)
    speed, strength = 1.3, 0.07
    wave_product = lambda t: (np.sqrt(2 / np.pi) * speed * t *
                              np.exp(-(speed * t) ** 2) * np.exp(1j * t))
    direct_amp = -1j * strength * gauss_integral(
        wave_product, -9 / speed, 9 / speed, nodes, weights)
    q = 1 / speed
    form_prob = (strength / speed)**2 * q**2 / 2 * np.exp(-q**2 / 2)
    report["5.34"] = {"A": strength, "c": speed, "direct_time_integral": abs(direct_amp)**2,
                       "form_factor_probability": form_prob}
    check("5.34 移动脉冲的时间积分与动量形状因子", abs(abs(direct_amp)**2 - form_prob), 1e-13)

    # 比较负时间制备+平台内两段积分与两个指数的表达式。负时尾截在−32/η。
    nodes, weights = leggauss(256)
    eta, freq_nm, end = 0.13, -0.8, 2.1
    for freq_mi in (1.7, 0.01, 0.0):
        before = gauss_integral(lambda t: np.exp((eta + 1j * freq_mi) * t),
                                -32 / eta, 0, nodes, weights)
        direct = gauss_integral(lambda t: np.exp(1j * freq_nm * t) *
                                (before + oscillatory_integral(freq_mi, t)),
                                0, end, nodes, weights)
        if freq_mi:
            closed = (oscillatory_integral(freq_nm + freq_mi, end)
                      - eta / (eta + 1j * freq_mi) * oscillatory_integral(freq_nm, end)) / (1j * freq_mi)
        else:
            # ∫_0^T exp(ivt)(1/η+t) dt，显式求t exp(ivt)的原函数。
            iv = 1j * freq_nm
            closed = ((np.exp(iv * end) - 1) / (iv * eta)
                      + np.exp(iv * end) * (end / iv - 1 / iv**2) + 1 / iv**2)
        report["5.38"].append({"eta": eta, "omega_mi": freq_mi, "omega_nm": freq_nm, "t": end,
                              "direct_kernel": complex_pair(direct), "closed_kernel": complex_pair(closed)})
        check(f"5.38 ω_mi={freq_mi}，平台两段积分与两个指数", abs(direct - closed), 1e-11)

    report["checks"] = checks
    report["all_passed"] = all(item["passed"] for item in checks)
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{sum(x['passed'] for x in checks)}/{len(checks)} checks passed")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
