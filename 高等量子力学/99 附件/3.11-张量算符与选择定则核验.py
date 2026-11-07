"""§3.11 教学补充：矩阵旋转、不可约张量、偶极积分和电四极矩。

原题 3.41—3.45 没有要求程序或绘图；本程序为独立交叉核验。
依赖 NumPy 与 Matplotlib，不依赖 SciPy/SymPy；输出位于脚本目录。
矩阵计算取 hbar=1，磁量子数降序，CG 采用 Condon–Shortley 相位。
CG 由最高权零空间和降算符生成；角积分由独立球谐递推和求积生成。
"""
from pathlib import Path
from math import factorial, pi, sqrt
from functools import lru_cache
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
CHECKS = {}
FONT = Path("C:/Windows/Fonts/msyh.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none",
                     "figure.facecolor": "white", "font.size": 11})


def check(name, value, tolerance=3e-12):
    error = float(np.max(np.abs(value)))
    old = CHECKS.get(name, {"max_abs_error": 0., "cases": 0})
    error = max(error, old["max_abs_error"])
    CHECKS[name] = {"max_abs_error": error, "tolerance": tolerance,
                    "cases": old["cases"]+1, "passed": error <= tolerance}
    if error > tolerance:
        raise AssertionError((name, error, tolerance))


def spin(j):
    ms = np.arange(j, -j-.25, -1.)
    plus = np.zeros((len(ms), len(ms)))
    for column, m in enumerate(ms):
        if column:
            plus[column-1, column] = sqrt((j-m)*(j+m+1))
    minus = plus.T
    return ms, (plus+minus)/2, (plus-minus)/(2j), np.diag(ms), plus, minus


@lru_cache(None)
def couple(j1, j2):
    ms1, _, _, _, p1, _ = spin(j1)
    ms2, _, _, _, p2, _ = spin(j2)
    pairs = [(float(m1), float(m2)) for m1 in ms1 for m2 in ms2]
    plus = np.kron(p1, np.eye(len(ms2)))+np.kron(np.eye(len(ms1)), p2)
    columns, labels = [], []
    for j in np.arange(j1+j2, abs(j1-j2)-.25, -1.):
        source = [i for i,(m1,m2) in enumerate(pairs) if abs(m1+m2-j)<1e-10]
        target = [i for i,(m1,m2) in enumerate(pairs) if abs(m1+m2-j-1)<1e-10]
        if not target:
            top = np.array([1.])
        else:
            _, singular, vh = np.linalg.svd(plus[np.ix_(target, source)], full_matrices=True)
            rank = int(np.sum(singular > 1e-12))
            if len(source)-rank != 1:
                raise AssertionError(("highest_weight_nullity", j1,j2,j))
            top = vh[rank, :].copy()
        top *= np.sign(next(value for value in top if abs(value)>1e-12))
        state = np.zeros(len(pairs))
        state[source] = top
        state /= np.linalg.norm(state)
        for m in np.arange(j, -j-.25, -1.):
            columns.append(state.copy())
            labels.append((float(j), float(m)))
            if m > -j+.25:
                state = plus.T@state/sqrt((j+m)*(j-m+1))
    return pairs, labels, np.column_stack(columns)


def cg(j,m,k,q,jp,mp):
    pairs, labels, matrix = couple(float(j), float(k))
    pair, label = (float(m),float(q)), (float(jp),float(mp))
    if pair not in pairs or label not in labels:
        return 0.
    return float(matrix[pairs.index(pair), labels.index(label)])


def rotation(j, alpha, beta, gamma):
    _, _, jy, jz, _, _ = spin(j)
    values, vectors = np.linalg.eigh(jy)
    middle = (vectors*np.exp(-1j*beta*values))@vectors.conj().T
    return np.diag(np.exp(-1j*alpha*np.diag(jz)))@middle@np.diag(
        np.exp(-1j*gamma*np.diag(jz)))


def spherical_components(x,y,z):
    return {1: -(x+1j*y)/sqrt(2), 0: z, -1: (x-1j*y)/sqrt(2)}


def coupled_products(u,v,k):
    return {q: sum(cg(1,q1,1,q2,k,q)*u[q1]@v[q2]
                   for q1 in [-1,0,1] for q2 in [-1,0,1])
            for q in range(-k,k+1)}


def tensor_checks():
    quadrupole_examples = []
    for j in [.5,1.,1.5,2.,2.5,3.]:
        ms,x,y,z,plus,minus = spin(j)
        vector = spherical_components(x,y,z)
        squared = x@x+y@y+z@z
        t = {k: coupled_products(vector,vector,k) for k in range(3)}
        check("CG_scalar_JJ",t[0][0]+squared/sqrt(3))
        for q in [-1,0,1]:
            check("CG_vector_JJ_noncommuting", t[1][q]+vector[q]/sqrt(2))
        cross = spherical_components(y@z-z@y,z@x-x@z,x@y-y@x)
        for q in [-1,0,1]:
            check("ordered_cross_JJ",cross[q]-1j*vector[q])
        check("quadrupole_zero_component",t[2][0]-(3*z@z-squared)/sqrt(6))
        check("quadrupole_minus_two",t[2][-2]-minus@minus/2)
        for k in range(3):
            for q in range(-k,k+1):
                value = t[k][q]
                check("tensor_Jz_commutator", z@value-value@z-q*value)
                expected = sqrt((k-q)*(k+q+1))*t[k][q+1] if q<k else np.zeros_like(value)
                check("tensor_Jplus_commutator",plus@value-value@plus-expected)
                for angles in [(.31,.78,-.46),(1.2,-.51,.82)]:
                    unitary = rotation(j,*angles)
                    dk = rotation(k,*angles)
                    index = int(k-q)
                    rhs = sum(dk[index,int(k-qp)].conj()*t[k][qp] for qp in range(-k,k+1))
                    check("finite_tensor_rotation",unitary.conj().T@value@unitary-rhs,1e-11)
        # Independent direct vector finite rotation around y (problem 3.42).
        beta=.73
        unitary=rotation(j,0,beta,0)
        check("Cartesian_y_rotation_x",unitary.conj().T@x@unitary-x*np.cos(beta)-z*np.sin(beta))
        check("Cartesian_y_rotation_z",unitary.conj().T@z@unitary-z*np.cos(beta)+x*np.sin(beta))
        if j==.5:
            for value in t[2].values():
                check("spin_half_rank_two_vanishes",value)
        else:
            # T^(2)=[J x J]^(2) is an explicit test operator for the universal ratio.
            moment = float((3*z@z-squared)[0,0].real)
            amplitudes = t[2][2]+t[2][-2]
            wanted = np.zeros(len(ms))
            wanted[2]=moment/sqrt(j*(2*j-1))
            check("quadrupole_stretched_state_ratio",amplitudes[:,0]-wanted)
            ratio = cg(j,j,2,-2,j,j-2)/cg(j,j,2,0,j,j)/sqrt(6)
            check("quadrupole_CG_ratio",ratio-1/sqrt(j*(2*j-1)))
            check("quadrupole_CG_q0_explicit",cg(j,j,2,0,j,j)-sqrt(j*(2*j-1)/((j+1)*(2*j+3))))
            check("quadrupole_CG_qminus2_explicit",cg(j,j,2,-2,j,j-2)-sqrt(6/((j+1)*(2*j+3))))
            quadrupole_examples.append({"j":j,"Q_hbar1":moment,
                "nonzero_m_prime":j-2,"amplitude":float(amplitudes[2,0].real),
                "amplitude_over_Q":float(amplitudes[2,0].real/moment)})
    return quadrupole_examples


def spherical(l,m,x,phi):
    if abs(m)>l:
        return np.zeros((len(x),len(phi)),dtype=complex)
    if m<0:
        return (-1)**m*spherical(l,-m,x,phi).conj()
    p=np.ones_like(x)
    for index in range(1,m+1):
        p *= -(2*index-1)*np.sqrt(np.maximum(0.,1-x*x))
    if l>m:
        prev,current=p,(2*m+1)*x*p
        for degree in range(m+2,l+1):
            prev,current=current,((2*degree-1)*x*current-(degree+m-1)*prev)/(degree-m)
        p=current
    return sqrt((2*l+1)/(4*pi)*factorial(l-m)/factorial(l+m))*p[:,None]*np.exp(1j*m*phi[None,:])


def dipole_coefficient(l,m,q,lp,mp):
    if mp!=m+q or abs(mp)>lp:
        return 0.
    if lp==l+1:
        den=(2*l+1)*(2*l+3)
        if q==0:
            return sqrt(((l+1)**2-m*m)/den)
        return sqrt((l+q*m+1)*(l+q*m+2)/(2*den))
    if lp==l-1:
        den=(2*l-1)*(2*l+1)
        if q==0:
            return sqrt((l*l-m*m)/den)
        return -sqrt((l-q*m)*(l-q*m-1)/(2*den))
    return 0.


def dipole_checks():
    x,weights=np.polynomial.legendre.leggauss(64)
    phi=np.arange(128)*(2*pi/128)
    ys={(l,m):spherical(l,m,x,phi) for l in range(6) for m in range(-l,l+1)}
    r={q:sqrt(4*pi/3)*ys[(1,q)] for q in [-1,0,1]}
    examples=[]
    for l in range(5):
        for m in range(-l,l+1):
            for q in [-1,0,1]:
                product=r[q]*ys[(l,m)]
                for lp in range(6):
                    for mp in range(-lp,lp+1):
                        angular=np.sum(weights[:,None]*ys[(lp,mp)].conj()*product)*(2*pi/len(phi))
                        analytic=dipole_coefficient(l,m,q,lp,mp)
                        coupled=sqrt((2*l+1)/(2*lp+1))*cg(l,0,1,0,lp,0)*cg(l,m,1,q,lp,mp)
                        check("dipole_integral_vs_analytic",angular-analytic)
                        check("dipole_CG_vs_analytic",coupled-analytic)
                        if (l,m,q,lp,mp) in [(0,0,1,1,1),(1,1,-1,0,0),(2,0,0,3,0),(2,0,0,2,0)]:
                            examples.append({"l":l,"m":m,"q":q,"l_prime":lp,"m_prime":mp,
                                "quadrature_real":float(angular.real),"analytic":float(analytic),
                                "CG":float(coupled)})
    return examples


def plot_dipole():
    fig,axes=plt.subplots(1,3,figsize=(13.8,6.4),constrained_layout=True)
    fig.suptitle("偶极角矩阵元：初态 l = 2 时的相对强度",fontsize=18)
    for ax,lp in zip(axes,[1,2,3]):
        data=np.array([[dipole_coefficient(2,m,q,lp,m+q)**2 for q in [-1,0,1]]
                       for m in [2,1,0,-1,-2]])
        img=ax.imshow(data,vmin=0,vmax=3/7,cmap="Blues",aspect="equal")
        for row,m in enumerate([2,1,0,-1,-2]):
            for col,q in enumerate([-1,0,1]):
                value=data[row,col]
                label="0" if value<1e-14 else f"{value:.4f}"
                ax.text(col,row,label,ha="center",va="center",fontsize=12,
                        color="white" if value>.23 else "#16324a")
        ax.set_title(f"末态 l′ = {lp}" + ("：奇偶性禁止" if lp==2 else ""),pad=15,fontsize=14)
        ax.set(xticks=[0,1,2],xticklabels=["−1","0","+1"],yticks=range(5),
               yticklabels=["+2","+1","0","−1","−2"],xlabel="张量分量 q",ylabel="初态 m")
        ax.set_xticks(np.arange(-.5,3,1),minor=True)
        ax.set_yticks(np.arange(-.5,5,1),minor=True)
        ax.grid(which="minor",color="white",lw=2)
        ax.tick_params(which="minor",bottom=False,left=False)
    cb=fig.colorbar(img,ax=axes,shrink=.78,pad=.03)
    cb.set_label(r"$|\langle l',m+q|\hat r_q|2,m\rangle|^2$",fontsize=12)
    fig.supxlabel("每格末态 m′ = m + q；超出 |m′| ≤ l′ 时为 0。\n"
                  "已除去径向矩阵元；图示为角向强度，不是不同径向通道的绝对跃迁率。",
                  fontsize=11)
    for ext in ["png","svg"]:
        fig.savefig(OUT/f"3.11-偶极选择定则角强度.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    quad=tensor_checks()
    dipole=dipole_checks()
    plot_dipole()
    result={"section":"3.11","purpose":"教学补充；原题未指定计算机或绘图",
        "units":"矩阵核验 hbar=1；偶极积分仅计算角部分",
        "conventions":{"CG":"Condon–Shortley","WE_denominator":"sqrt(2j_prime+1)",
            "rotation":"D=exp(-i alpha Jz) exp(-i beta Jy) exp(-i gamma Jz)",
            "CG_scalar":"-U dot V / sqrt(3)","CG_rank_one":"i (U cross V)_q / sqrt(2)",
            "textbook_3_469_scalar":"-U dot V / 3",
            "textbook_3_469_rank_one":"(U cross V)_q / (i sqrt(2))"},
        "quadrature":{"Gauss_Legendre_nodes":64,"phi_nodes":128,"initial_l":[0,1,2,3,4],"final_l":[0,1,2,3,4,5]},
        "checks":CHECKS,"all_passed":all(item["passed"] for item in CHECKS.values()),
        "total_checks":sum(item["cases"] for item in CHECKS.values()),
        "dipole_examples":dipole,"quadrupole_examples":quad,
        "figures":["3.11-偶极选择定则角强度.png","3.11-偶极选择定则角强度.svg"]}
    path=OUT/"3.11-张量算符与选择定则核验结果.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"all_passed":result["all_passed"],"total_checks":result["total_checks"],
        "largest_absolute_error":max(x["max_abs_error"] for x in CHECKS.values()),
        "output":str(path)},ensure_ascii=False))


if __name__=="__main__":
    main()
