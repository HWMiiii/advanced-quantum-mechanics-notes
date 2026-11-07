"""复算《现代量子力学》第5章§5.4的两个教材例子与方阱重叠。

运行后在本目录写入核验 JSON、参数扫描 CSV 与一张 PNG。
依赖 NumPy、Matplotlib。解析能量由独立数值积分检验，重叠由积分及余弦级数双算。
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parent.parent
local_deps = PROJECT / 'tmp' / 'section3_1_deps'
if sys.version_info[:2] == (3, 12) and local_deps.is_dir():
    sys.path.insert(0, str(local_deps))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / 'tmp' / 'section5_4_variation' / 'mpl'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

PI = math.pi
OPTIMAL = (1 + math.sqrt(6)) / 2
nodes, weights = np.polynomial.legendre.leggauss(96)
unit_nodes = (nodes + 1) / 2
unit_weights = weights / 2


def box_norm(lam):
    """Integral of the squared dimensionless shape on [-1,1]."""
    return 4 * lam**2 / ((lam + 1) * (2 * lam + 1))


def box_energy_ratio(lam):
    """Rayleigh quotient divided by E0; finite for lambda > 1/2."""
    return 2 * (lam + 1) * (2 * lam + 1) / ((2 * lam - 1) * PI**2)


def box_integrals(lam):
    """Independently integrate norm, gradient norm and ground overlap.

    On [0,1] use y=t^4 to resolve the possible derivative singularity at zero.
    """
    y = unit_nodes**4
    dy = 4 * unit_nodes**3
    shape = 1 - y**lam
    derivative = -lam * y**(lam - 1)
    norm = 2 * np.dot(unit_weights, dy * shape**2)
    grad_norm = 2 * np.dot(unit_weights, dy * derivative**2)
    ground_overlap = 2 * np.dot(unit_weights, dy * shape * np.cos(PI * y / 2)) / math.sqrt(norm)
    return float(norm), float(grad_norm), float(ground_overlap)


def overlap_series(lam):
    """Integrate the Taylor series for cos(pi*y/2) term by term."""
    terms = []
    for j in range(18):
        coefficient = (-1)**j * (PI / 2)**(2 * j) / math.factorial(2 * j)
        moment = lam / ((2 * j + 1) * (2 * j + 1 + lam))
        terms.append(coefficient * moment)
    return 2 * math.fsum(terms) / math.sqrt(box_norm(lam))


def sample(lam):
    norm, grad_norm, overlap = box_integrals(lam)
    energy = box_energy_ratio(lam)
    fidelity_bound = 1 - (energy - 1) / 8
    angle_bound = math.degrees(math.acos(math.sqrt(fidelity_bound)))
    return {
        'lambda': lam, 'norm_integral': norm, 'gradient_integral': grad_norm,
        'analytic_energy_over_E0': energy, 'quadrature_energy_over_E0': 4 * grad_norm / (PI**2 * norm),
        'relative_energy_excess_percent': 100 * (energy - 1),
        'direct_overlap': overlap, 'series_overlap': overlap_series(lam),
        'exact_fidelity': overlap**2, 'energy_based_fidelity_lower_bound': fidelity_bound,
        'actual_angle_deg': math.degrees(math.acos(overlap)), 'angle_upper_bound_deg': angle_bound,
    }


def check(name, residual, tolerance):
    value = float(abs(residual))
    return {'item': name, 'residual': value, 'tolerance': tolerance, 'passed': value < tolerance}


def check_hydrogen():
    # a0=Eh=1. u=2r/a converts exp(-2r/a) to the Laguerre weight exp(-u).
    u, w = np.polynomial.laguerre.laggauss(64)
    entries, checks = [], []
    for a in (0.6, 0.9, 1.0, 1.2, 2.0):
        norm = float(np.dot(w, u**2 / 2))
        mean_inverse_r = float(np.dot(w, u / a))
        kinetic = norm / (2 * a**2)
        energy = kinetic - mean_inverse_r
        exact_excess = (1 / a - 1)**2 / 2
        entries.append({'a_over_a0': a, 'norm': norm, 'inverse_r_times_a0': mean_inverse_r,
                        'kinetic_over_Eh': kinetic, 'potential_over_Eh': -mean_inverse_r,
                        'energy_over_Eh': energy, 'excess_over_Eh': exact_excess})
        checks += [check(f'氢归一化 a/a0={a:g}', norm - 1, 1e-11),
                   check(f'氢逆半径积分 a/a0={a:g}', mean_inverse_r - 1/a, 1e-11),
                   check(f'氢能量配方 a/a0={a:g}', energy + .5 - exact_excess, 1e-11)]
    return entries, checks


def draw(samples):
    for font_path in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
        if Path(font_path).is_file():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = [font_manager.FontProperties(fname=font_path).get_name(), 'DejaVu Sans']
            break
    plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False})
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.7, 4.0), layout='constrained')
    x = [row['lambda'] for row in samples]
    left.plot(x, [row['relative_energy_excess_percent'] for row in samples],
              color='#245995', lw=2.1)
    left.set(xlabel='形状参数 λ', ylabel='能量相对偏高（%）', title='能量上界随试探形状变化', xlim=(1, 3))
    left.set_yscale('log')
    left.set_yticks([.3, 1, 3, 10, 20], labels=['0.3', '1', '3', '10', '20'])
    left.set_ylim(.23, 28)
    for lam, label, offset in [(OPTIMAL, '最优  0.298%', (0, 12)), (2., '抛物线  1.32%', (9, 8))]:
        row = sample(lam)
        left.scatter([lam], [row['relative_energy_excess_percent']], color='#b44b2d', zorder=4)
        left.annotate(label, (lam, row['relative_energy_excess_percent']), xytext=offset,
                      textcoords='offset points', fontsize=9)
    right.plot(x, [row['angle_upper_bound_deg'] for row in samples], '--', label='由能量和偶宇称得到的上界',
               color='#b4612a', lw=2.0)
    right.plot(x, [row['actual_angle_deg'] for row in samples], label='与精确基态的实际夹角',
               color='#168075', lw=2.1)
    right.set(xlabel='形状参数 λ', ylabel='态矢夹角（度）', title='夹角上界与直接重叠积分', xlim=(1, 3))
    right.legend(frameon=False, loc='upper center', fontsize=9)
    for ax in (left, right):
        ax.axvline(OPTIMAL, ls=':', color='#7d8892', lw=.9)
        ax.grid(axis='y', alpha=.2)
    figure.savefig(OUTPUT / '5.4-方阱变分上界与态夹角.png', dpi=190, bbox_inches='tight')
    plt.close(figure)


def main():
    hydrogen, checks = check_hydrogen()
    selected = [sample(lam) for lam in (.75, 1., 1.25, OPTIMAL, 2., 3.)]
    for row in selected:
        lam = row['lambda']
        checks.extend([
            check(f'阱范数积分 λ={lam:.6f}', row['norm_integral'] - box_norm(lam), 1e-12),
            check(f'阱能量积分 λ={lam:.6f}', row['quadrature_energy_over_E0'] - row['analytic_energy_over_E0'], 1e-12),
            check(f'余弦级数与直接积分 λ={lam:.6f}', row['series_overlap'] - row['direct_overlap'], 1e-12),
            check(f'谱隙给出的重叠下界 λ={lam:.6f}',
                  max(0., row['energy_based_fidelity_lower_bound'] - row['exact_fidelity']), 1e-12)])
    checks.append(check('λ=2时重叠解析核对', overlap_series(2) - 8*math.sqrt(15)/PI**3, 1e-12))
    checks.append(check('极小点满足导数方程', 4*OPTIMAL**2 - 4*OPTIMAL - 5, 1e-12))
    checks.append(check('闭式最优能量', box_energy_ratio(OPTIMAL) - (5 + 2*math.sqrt(6))/PI**2, 1e-12))
    odd_shape = nodes * (1 - nodes**2)
    odd_norm = float(np.dot(weights, odd_shape**2))
    odd_kinetic = float(np.dot(weights, (1 - 3*nodes**2)**2))
    checks.append(check('奇宇称试探态范数', odd_norm - 16/105, 1e-12))
    checks.append(check('奇宇称试探态的梯度平方积分', odd_kinetic - 8/5, 1e-12))
    samples = [sample(float(lam)) for lam in np.linspace(1., 3., 401)]
    with (OUTPUT / '5.4-方阱变分参数扫描.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)
    result = {'source': '§5.4 PDF332—335／印刷310—313；实际重叠积分与图为教学补充。',
              'units': '阱能量用 E0，角度用度数。氢的能量单位为 Hartree，长度单位为 a0。',
              'integration': '氢64点Gauss–Laguerre；方阱96点Gauss–Legendre、y=t^4。重叠另以余弦级数前18项交叉核验。',
              'hydrogen': hydrogen, 'box_optimum': sample(OPTIMAL), 'box_parabola': sample(2.),
              'box_samples': selected, 'odd_trial': {'norm_squared': odd_norm, 'derivative_norm_squared': odd_kinetic,
                  'rayleigh_over_first_excited_energy': odd_kinetic / (PI**2 * odd_norm)},
              'checks': checks, 'passed': all(item['passed'] for item in checks)}
    (OUTPUT / '5.4-方阱变分与重叠核验.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'checks': len(checks), 'failed': [x for x in checks if not x['passed']],
                      'optimum': result['box_optimum'], 'parabola': result['box_parabola']}, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise RuntimeError('数值核验未通过。')
    draw(samples)


if __name__ == '__main__':
    main()
