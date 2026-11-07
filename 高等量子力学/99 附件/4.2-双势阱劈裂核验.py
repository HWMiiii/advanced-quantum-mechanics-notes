"""教材4.6的教学数值核验；原题没有指定数值。

单位 hbar=m=b=1。解析匹配方程独立二分求根；没有有限差分网格误差。
even在两阱为正，odd在右阱为正。精确分段解使用解析归一化，
CSV供科学作图，不改变教材PDF。只需Python标准库。
"""
from pathlib import Path
import math
import json
import csv

OUT = Path(__file__).resolve().parent
V0, B = 100.0, 1.0


def bisect_root(fun, lo, hi):
    flo = fun(lo)
    assert flo * fun(hi) < 0
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        fm = fun(mid)
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi = mid
    root = (lo + hi) / 2
    return root, hi - lo, abs(fun(root))


def solve(a, parity):
    def f(k):
        kap = math.sqrt(2 * V0 - k * k)
        q = 1 if parity == 'isolated' else (math.tanh(kap * a) if parity == 'even' else 1 / math.tanh(kap * a))
        return k / math.tan(k * B) + kap * q
    k, width, residual = bisect_root(f, math.pi / (2 * B) + 1e-12, math.pi / B - 1e-12)
    return {'E': k * k / 2, 'k': k, 'kappa': math.sqrt(2 * V0 - k * k),
            'k_bracket_width': width, 'matching_residual': residual}


def waveform(a, sol, parity):
    k, kap = sol['k'], sol['kappa']
    sign = 1 if parity == 'even' else -1
    f = math.cosh if parity == 'even' else math.sinh
    denom = f(kap * a)
    norm2 = B - math.sin(2 * k * B) / (2 * k)
    norm2 += math.sin(k * B) ** 2 / denom ** 2 * (sign * a + math.sinh(2 * kap * a) / (2 * kap))
    normalizer = 1 / math.sqrt(norm2)
    def psi(x):
        if abs(x) > a + B:
            return 0.0
        if abs(x) < a:
            return normalizer * math.sin(k * B) * f(kap * x) / denom
        value = normalizer * math.sin(k * (a + B - abs(x)))
        return value if parity == 'even' or x >= 0 else -value
    return psi, normalizer


def trapz(values, dx):
    return dx * (sum(values) - (values[0] + values[-1]) / 2)


cases = []
for a in (0.2, 0.4, 0.6):
    even, odd, isolated = (solve(a, p) for p in ('even', 'odd', 'isolated'))
    k, kap = isolated['k'], isolated['kappa']
    gap = odd['E'] - even['E']
    improved = 4 * kap * k * k / ((k * k + kap * kap) * (B + 1 / kap)) * math.exp(-2 * kap * a)
    k0 = math.pi / B
    kap0 = math.sqrt(2 * V0 - k0 * k0)
    high_barrier = 4 * k0 * k0 / (kap0 * B) * math.exp(-2 * kap0 * a)
    cases.append({'a': a, 'b': B, 'V0': V0, 'even': even, 'odd': odd,
                  'isolated': isolated, 'exact_gap': gap, 'leading_tunneling_gap': improved,
                  'leading_relative_error': (improved - gap) / gap,
                  'strong_barrier_gap': high_barrier,
                  'strong_barrier_relative_error': (high_barrier - gap) / gap})
    assert odd['E'] > even['E']
    assert max(even['matching_residual'], odd['matching_residual']) < 1e-10

a = 0.4
case = cases[1]
even_psi, ne = waveform(a, case['even'], 'even')
odd_psi, no = waveform(a, case['odd'], 'odd')
count = 5600
dx = 2 * (a + B) / count
xs = [-(a + B) + i * dx for i in range(count + 1)]
even_values = [even_psi(x) for x in xs]
odd_values = [odd_psi(x) for x in xs]
csv_path = OUT / '4.2-双势阱本征态与局域态.csv'
with csv_path.open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'V', 'psi_even', 'psi_odd', 'psi_right', 'psi_left'])
    for x, ev, od in zip(xs, even_values, odd_values):
        writer.writerow([x, V0 if abs(x) < a else 0, ev, od,
                         (ev + od) / math.sqrt(2), (ev - od) / math.sqrt(2)])

half = count // 2
checks = {
    'even_norm_trapz': trapz([v * v for v in even_values], dx),
    'odd_norm_trapz': trapz([v * v for v in odd_values], dx),
    'even_odd_overlap_trapz': trapz([e * o for e, o in zip(even_values, odd_values)], dx),
    'even_odd_right_half_overlap': trapz([e * o for e, o in zip(even_values[half:], odd_values[half:])], dx),
    'even_normalizer': ne, 'odd_normalizer': no,
    'csv_points': count + 1, 'csv_dx': dx,
    'outer_boundary_abs_max': max(abs(even_values[0]), abs(even_values[-1]), abs(odd_values[0]), abs(odd_values[-1])),
}
assert abs(checks['even_norm_trapz'] - 1) < 1e-7
assert abs(checks['odd_norm_trapz'] - 1) < 1e-7
assert abs(checks['even_odd_overlap_trapz']) < 1e-10

result = {'source': 'Exercise4.6, PDF303/printed281',
          'note': 'Explicit teaching parameters; no numerical parameters are prescribed by the textbook.',
          'units': 'hbar=m=b=1', 'cases': cases, 'waveform_case': 'a=0.4, b=1, V0=100',
          'phase_convention': 'even positive; odd positive in right well',
          'normalization': 'analytic piecewise integrals', 'waveform_checks': checks,
          'csv_file': csv_path.name}
output = OUT / '4.2-双势阱劈裂核验.json'
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
