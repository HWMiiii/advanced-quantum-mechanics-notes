"""§4.1 教学补充核验。不是教材原题要求，也不替代正文证明。

取 hbar=1。第一部分以有限字典表示 x^a y^b z^c r^s，按乘积法则
解析求导，检验所列算符恒等式作用在有限测试函数集上的残差。
它不是有限差分，亦不是对无限维算符的穷尽证明。
第二部分在 C^2 tensor C^2 构造 n=2 的 I=K=1/2 表示。
依赖：Python、NumPy。结果保存于本脚本同目录。
"""
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parent
CHECKS = {}
MASS, KAPPA = 1.7, 2.3


def combine(*terms):
    result = {}
    for scale, poly in terms:
        for key, val in poly.items():
            result[key] = result.get(key, 0) + scale * val
    return {key: val for key, val in result.items() if val != 0}


def multiply(f, axis=None, radial=0):
    result = {}
    for key, val in f.items():
        new = list(key)
        if axis is not None:
            new[axis] += 1
        new[3] += radial
        result[tuple(new)] = result.get(tuple(new), 0) + val
    return result


def derivative(f, axis):
    result = {}
    for key, val in f.items():
        power, radial = key[axis], key[3]
        if power:
            new = list(key)
            new[axis] -= 1
            new = tuple(new)
            result[new] = result.get(new, 0) + power * val
        if radial:
            new = list(key)
            new[axis] += 1
            new[3] -= 2
            new = tuple(new)
            result[new] = result.get(new, 0) + radial * val
    return result


def p(axis, f):
    return combine((-1j, derivative(f, axis)))


def p2(f):
    return combine(*[(1, p(i, p(i, f))) for i in range(3)])


def L(axis, f):
    j, k = (axis + 1) % 3, (axis + 2) % 3
    return combine((1, multiply(p(k, f), j)), (-1, multiply(p(j, f), k)))


def C(axis, f):
    j, k = (axis + 1) % 3, (axis + 2) % 3
    return combine((1, p(j, L(k, f))), (-1, p(k, L(j, f))), (-1j, p(axis, f)))


def nvec(axis, f):
    return multiply(f, axis, -1)


def M(axis, f):
    return combine((1 / MASS, C(axis, f)), (-KAPPA, nvec(axis, f)))


def H(f):
    return combine((1 / (2 * MASS), p2(f)), (-KAPPA, multiply(f, radial=-1)))


def L2(f):
    return combine(*[(1, L(i, L(i, f))) for i in range(3)])


def evaluate(f, point):
    x, y, z = point
    r = np.linalg.norm(point)
    return sum(v * x**a * y**b * z**c * r**s for (a, b, c, s), v in f.items())


POINTS = [(1.2, 0.7, -0.4), (-0.8, 1.5, 0.9), (0.4, -0.6, 1.7)]
FUNCTIONS = [{key: 1} for key in [(0,0,0,0), (1,0,0,0), (1,1,0,0),
             (2,0,1,0), (0,1,2,1), (1,0,1,-1), (0,2,0,-3), (2,1,1,2)]]


def record(name, residual, tol=2e-9):
    val = float(np.max(np.abs(residual)))
    old = CHECKS.get(name, {'cases': 0, 'max_abs_error': 0.})
    val = max(old['max_abs_error'], val)
    CHECKS[name] = {'cases': old['cases'] + 1, 'max_abs_error': val,
                    'tolerance': tol, 'passed': val <= tol}
    if val > tol:
        raise AssertionError((name, val, tol))


def check_function(name, f):
    record(name, [evaluate(f, pt) for pt in POINTS])


for f in FUNCTIONS:
    check_function('C_squared', combine(
        (1, combine(*[(1, C(i, C(i, f))) for i in range(3)])),
        (-1, p2(combine((1, L2(f)), (1, f))))))
    check_function('symmetrized_C_dot_n', combine(
        (1, combine(*[(1, C(i, nvec(i, f))) for i in range(3)],
                    *[(1, nvec(i, C(i, f))) for i in range(3)])),
        (-2, multiply(combine((1, L2(f)), (1, f)), radial=-1))))
    check_function('L_dot_M', combine(*[(1, L(i, M(i, f))) for i in range(3)]))
    check_function('M_dot_L', combine(*[(1, M(i, L(i, f))) for i in range(3)]))
    check_function('M_squared', combine(
        (1, combine(*[(1, M(i, M(i, f))) for i in range(3)])),
        (-2 / MASS, H(combine((1, L2(f)), (1, f)))), (-KAPPA**2, f)))
    for i in range(3):
        check_function('M_H_commutator', combine((1, M(i, H(f))), (-1, H(M(i, f)))))
        j, k = (i + 1) % 3, (i + 2) % 3
        check_function('M_M_commutator', combine((1, M(i, M(j, f))), (-1, M(j, M(i, f))),
                       (2j / MASS, H(L(k, f)))))

# n=2: an exact four-dimensional representation, floating-point arithmetic only.
eye2, eye4 = np.eye(2), np.eye(4)
sigma = [np.array([[0,1],[1,0]],complex), np.array([[0,-1j],[1j,0]],complex),
         np.diag([1.,-1.])]
I = [np.kron(s / 2, eye2) for s in sigma]
K = [np.kron(eye2, s / 2) for s in sigma]
Lv = [a + b for a, b in zip(I,K)]
Nv = [a - b for a, b in zip(I,K)]
comm = lambda a,b: a @ b - b @ a
for i in range(3):
    j, k = (i + 1) % 3, (i + 2) % 3
    record('finite_I_algebra', comm(I[i],I[j])-1j*I[k])
    record('finite_K_algebra', comm(K[i],K[j])-1j*K[k])
    record('finite_N_algebra', comm(Nv[i],Nv[j])-1j*Lv[k])
    for j in range(3):
        record('finite_I_K_commute', comm(I[i],K[j]))
LL = sum(a @ a for a in Lv)
NN = sum(a @ a for a in Nv)
record('finite_Casimir', LL+NN-3*eye4)
record('finite_L_dot_N', sum(a@b for a,b in zip(Lv,Nv)))
singlet = np.array([0,1,-1,0]) / np.sqrt(2)
triplet0 = np.array([0,1,1,0]) / np.sqrt(2)
record('N_z_maps_l0_to_l1', Nv[2] @ singlet - triplet0)
record('singlet_L_squared_zero', LL @ singlet)
record('triplet_L_squared_two', LL @ triplet0 - 2*triplet0)
result = {'status': 'PASS', 'hbar': 1, 'mass': MASS, 'kappa': KAPPA,
          'test_function_count': len(FUNCTIONS), 'test_points': POINTS,
          'checks': CHECKS, 'n2_L_squared_eigenvalues': np.linalg.eigvalsh(LL).tolist(),
          'scope': 'Finite test functions and finite n=2 representation; not an exhaustive proof.'}
(OUT/'4.1-对称性与SO4代数核验结果.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'status': result['status'], 'groups':len(CHECKS),
      'max_abs_error':max(v['max_abs_error'] for v in CHECKS.values())}))
