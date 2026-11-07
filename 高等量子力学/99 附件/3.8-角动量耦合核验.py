"""Sakurai 中文第3版 §3.8 的独立教学核验与可重算图。

本节所配原题没有指定计算机作图任务；本程序是教学补充。
约定 ħ=1；C 的行是乘积基，列是耦合基；两种磁量子数均降序。
每个 j 的最高权态中，最大 m1 的非零系数取正，其余由 J_- 确定。
依赖 NumPy、Matplotlib，无需 SciPy。输出到本脚本所在目录。
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
FONT = Path("C:/Windows/Fonts/msyh.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none",
                     "figure.facecolor": "white", "font.size": 11})
CHECKS = {}


def check(name, error, tolerance=2e-12):
    error = float(np.max(np.abs(error)))
    old = CHECKS.get(name, {"max_abs_error": 0., "cases": 0})
    CHECKS[name] = {"max_abs_error": max(error, old["max_abs_error"]),
                    "tolerance": tolerance, "cases": old["cases"]+1,
                    "passed": max(error, old["max_abs_error"]) <= tolerance}
    if error > tolerance:
        raise AssertionError((name, error, tolerance))


def spin(j):
    ms = np.arange(j, -j-0.25, -1.)
    jp = np.zeros((len(ms), len(ms)))
    for column, m in enumerate(ms):
        if column:
            jp[column-1, column] = sqrt((j-m)*(j+m+1))
    jm = jp.T
    return ms, (jp+jm)/2, (jp-jm)/(2j), np.diag(ms), jp, jm


@lru_cache(None)
def couple(j1, j2):
    ms1, x1, y1, z1, p1, n1 = spin(j1)
    ms2, x2, y2, z2, p2, n2 = spin(j2)
    i1, i2 = np.eye(len(ms1)), np.eye(len(ms2))
    pairs = [(float(m1), float(m2)) for m1 in ms1 for m2 in ms2]
    jp = np.kron(p1, i2)+np.kron(i1, p2)
    jm = jp.T
    jz = np.kron(z1, i2)+np.kron(i1, z2)
    j2op = jz@jz+(jp@jm+jm@jp)/2
    columns, labels = [], []
    for j in np.arange(j1+j2, abs(j1-j2)-.25, -1.):
        source = [i for i,(m1,m2) in enumerate(pairs) if abs(m1+m2-j)<1e-10]
        target = [i for i,(m1,m2) in enumerate(pairs) if abs(m1+m2-j-1)<1e-10]
        block = jp[np.ix_(target, source)]
        if not target:
            top = np.array([1.])
        else:
            _, singular, vh = np.linalg.svd(block, full_matrices=True)
            rank = int(np.sum(singular > 1e-12))
            if len(source)-rank != 1:
                raise AssertionError(("Highest-weight nullity", j1,j2,j))
            top = vh[rank, :]
        first = next(value for value in top if abs(value)>1e-12)
        top *= np.sign(first)
        state = np.zeros(len(pairs))
        state[source] = top
        state /= np.linalg.norm(state)
        for m in np.arange(j, -j-.25, -1.):
            columns.append(state.copy())
            labels.append((float(j),float(m)))
            if m > -j+.25:
                state = jm@state/sqrt((j+m)*(j-m+1))
    c = np.column_stack(columns)
    return pairs, labels, c, jp, jm, jz, j2op


def rotation(j, angles):
    _, _, jy, jz, _, _ = spin(j)
    alpha, beta, gamma = angles
    values, vectors = np.linalg.eigh(jy)
    middle = (vectors*np.exp(-1j*beta*values))@vectors.conj().T
    return np.diag(np.exp(-1j*alpha*np.diag(jz)))@middle@np.diag(
        np.exp(-1j*gamma*np.diag(jz)))


def check_coupling():
    cases = [(0.,.5),(.5,.5),(1.,.5),(1.,1.),(1.5,1.),(2.,1.5),(2.,2.),(3.,2.)]
    angles_list = [(.37,.81,-.46),(1.2,.25,2.1)]
    for j1,j2 in cases:
        pairs, labels, c, jp, jm, jz, j2op = couple(j1,j2)
        check("CG_orthogonality", c.T@c-np.eye(len(c)))
        check("J2_eigenvectors", j2op@c-c*np.array([j*(j+1) for j,m in labels]))
        check("Jz_eigenvectors", jz@c-c*np.array([m for j,m in labels]))
        for col,(j,m) in enumerate(labels):
            expected = np.zeros(len(c))
            if m < j-.25:
                expected = sqrt((j-m)*(j+m+1))*c[:,labels.index((j,m+1))]
            check("Jplus_ladder",jp@c[:,col]-expected)
        for angles in angles_list:
            product = np.kron(rotation(j1,angles),rotation(j2,angles))
            expected = np.zeros(product.shape,dtype=complex)
            offset = 0
            for j in np.arange(j1+j2,abs(j1-j2)-.25,-1.):
                block = rotation(j,angles)
                expected[offset:offset+len(block),offset:offset+len(block)] = block
                offset += len(block)
            check("rotation_direct_sum",c.T@product@c-expected,8e-12)
    # Analytic 1 x 1 coefficients are entered separately from null-space construction.
    pairs,labels,c,*_ = couple(1.,1.)
    vectors = {
        (2.,2.): {(1.,1.):1.},
        (2.,1.): {(1.,0.):1/sqrt(2),(0.,1.):1/sqrt(2)},
        (2.,0.): {(1.,-1.):1/sqrt(6),(0.,0.):2/sqrt(6),(-1.,1.):1/sqrt(6)},
        (2.,-1.): {(0.,-1.):1/sqrt(2),(-1.,0.):1/sqrt(2)},
        (2.,-2.): {(-1.,-1.):1.},
        (1.,1.): {(1.,0.):1/sqrt(2),(0.,1.):-1/sqrt(2)},
        (1.,0.): {(1.,-1.):1/sqrt(2),(-1.,1.):-1/sqrt(2)},
        (1.,-1.): {(0.,-1.):1/sqrt(2),(-1.,0.):-1/sqrt(2)},
        (0.,0.): {(1.,-1.):1/sqrt(3),(0.,0.):-1/sqrt(3),(-1.,1.):1/sqrt(3)}
    }
    analytic = np.array([[vectors[label].get(pair,0.) for label in labels] for pair in pairs])
    check("analytic_1x1_CG", c-analytic)
    return cases


def spherical(l,m,x,phi):
    """Condon-Shortley Y_l^m; x=cos(theta). Arrays use separate broadcast axes."""
    if abs(m)>l:
        return np.zeros((len(x),len(phi)),dtype=complex)
    if m<0:
        return (-1)**m*spherical(l,-m,x,phi).conj()
    p = np.ones_like(x)
    for index in range(1,m+1):
        p *= -(2*index-1)*np.sqrt(np.maximum(0.,1-x*x))
    if l>m:
        prev, current = p, (2*m+1)*x*p
        for degree in range(m+2,l+1):
            prev,current = current,((2*degree-1)*x*current-(degree+m-1)*prev)/(degree-m)
        p = current
    factor = sqrt((2*l+1)/(4*pi)*factorial(l-m)/factorial(l+m))
    return factor*p[:,None]*np.exp(1j*m*phi[None,:])


def sphere_checks():
    x, weights = np.polynomial.legendre.leggauss(64)
    phi = np.arange(128)*(2*pi/128)
    ys = {(l,m): spherical(l,m,x,phi) for l in range(7) for m in range(-l,l+1)}
    def integral(f):
        return np.sum(weights[:,None]*f)*(2*pi/len(phi))
    for l in range(6):
        for sign in [1,-1]:
            j = l+sign*.5
            if j<0:
                continue
            pairs, labels, c, *_ = couple(float(l),.5)
            for m in np.arange(j,-j-.25,-1.):
                if sign==1:
                    a=sqrt((l+m+.5)/(2*l+1)); b=sqrt((l-m+.5)/(2*l+1))
                    eigen_ls=l/2
                else:
                    a=-sqrt((l-m+.5)/(2*l+1)); b=sqrt((l+m+.5)/(2*l+1))
                    eigen_ls=-(l+1)/2
                vector=np.zeros(len(pairs))
                for pair,amplitude in [((m-.5,.5),a),((m+.5,-.5),b)]:
                    if pair in pairs:
                        vector[pairs.index(pair)] = amplitude
                check("analytic_l_spin_half_CG",c[:,labels.index((j,float(m)))]-vector)
                f=ys[(l,int(m-.5))] if abs(m-.5)<=l else np.zeros_like(ys[(0,0)])
                g=ys[(l,int(m+.5))] if abs(m+.5)<=l else np.zeros_like(ys[(0,0)])
                check("spinor_sphere_normalization",integral(abs(a*f)**2+abs(b*g)**2)-1)
                coupling=sqrt(max(0.,(l+.5)**2-m*m))
                ls=.5*np.array([[m-.5,coupling],[coupling,-m-.5]])
                v=np.array([a,b])
                check("spinor_LdotS_eigenvalue",ls@v-eigen_ls*v)
                j2matrix=(l*(l+1)+.75)*np.eye(2)+2*ls
                check("spinor_J2_eigenvalue",j2matrix@v-j*(j+1)*v)
    examples=[]
    for l1 in range(3):
        for l2 in range(3):
            pairs,labels,c,*_=couple(float(l1),float(l2))
            for l in range(5):
                c00=c[pairs.index((0.,0.)),labels.index((float(l),0.))] if (float(l),0.) in labels else 0.
                for m1 in range(-l1,l1+1):
                    for m2 in range(-l2,l2+1):
                        for m in [m1+m2,m1+m2+1]:
                            if abs(m)>l:
                                continue
                            actual=integral(ys[(l,m)].conj()*ys[(l1,m1)]*ys[(l2,m2)])
                            cm=c[pairs.index((float(m1),float(m2))),labels.index((float(l),float(m)))] if (float(l),float(m)) in labels else 0.
                            predicted=sqrt((2*l1+1)*(2*l2+1)/(4*pi*(2*l+1)))*c00*cm
                            check("Gaunt_sphere_quadrature",actual-predicted,3e-12)
                            if (l1,m1,l2,m2,l,m) in [(1,0,1,0,0,0),(1,0,1,0,2,0),(1,0,1,0,1,0)]:
                                examples.append({"l1":l1,"m1":m1,"l2":l2,"m2":m2,"l":l,"m":m,
                                                 "numerical_real":float(actual.real),"predicted":float(predicted)})
    return examples


def plot():
    pairs,labels,c,*_=couple(1.,1.)
    fig,(ax,heat)=plt.subplots(1,2,figsize=(15.2,7.1),gridspec_kw={"width_ratios":[.9,1.45]},
                                constrained_layout=True)
    fig.suptitle(r"$1\otimes1$：乘积态格点与 Clebsch–Gordan 换基",fontsize=17)
    ax.set_title("固定 M 的态数决定各 j 多重态数",pad=16)
    for total in range(-2,3):
        grid=np.linspace(-1.45,1.45,150)
        valid=(-1.45<=total-grid)&(total-grid<=1.45)
        ax.plot(grid[valid],(total-grid)[valid],ls="--",lw=1,color="#c8d2de",zorder=0)
    for m1,m2 in pairs:
        ax.scatter(m1,m2,s=610,c=[plt.cm.coolwarm((m1+m2+2)/4)],edgecolor="#536579",lw=1.1)
        ax.text(m1,m2,f"M={int(m1+m2)}",ha="center",va="center",fontsize=11,
                color="white" if abs(m1+m2)==2 else "#182433")
    ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),xticks=[-1,0,1],yticks=[-1,0,1],
           xlabel=r"$m_1$",ylabel=r"$m_2$")
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    ax.text(.5,-.19,r"$M=-2,-1,0,1,2$："+r"$N_M=1,2,3,2,1$"+"\n"
            +r"$n_j=N_j-N_{j+1}$："+r"$j=2,1,0$ 各出现一次"+"\n"
            +r"维数：$3\times3=5+3+1$",
            transform=ax.transAxes,ha="center",va="top",fontsize=11,linespacing=1.65)
    im=heat.imshow(c,cmap="RdBu_r",vmin=-1,vmax=1,aspect="equal")
    heat.set_title(r"$C_{(m_1,m_2),(j,M)}=\langle m_1,m_2|j,M\rangle$",pad=16)
    lab=lambda pair:f"({int(pair[0])},{int(pair[1])})"
    heat.set(xticks=range(9),yticks=range(9),xticklabels=[lab(q) for q in labels],
             yticklabels=[lab(q) for q in pairs],xlabel="耦合基列 (j, M)",ylabel=r"乘积基行 $(m_1,m_2)$")
    heat.tick_params(labelsize=10)
    heat.set_xticks(np.arange(-.5,9,1),minor=True)
    heat.set_yticks(np.arange(-.5,9,1),minor=True)
    heat.grid(which="minor",color="white",linewidth=.8)
    heat.tick_params(which="minor",bottom=False,left=False)
    for pos in [4.5,7.5]:
        heat.axvline(pos,color="#15233a",lw=2)
    for row in range(9):
        for col in range(9):
            if abs(c[row,col])>1e-12:
                heat.text(col,row,f"{c[row,col]:.3f}",ha="center",va="center",fontsize=9,
                          color="white" if abs(c[row,col])>.65 else "#182433")
    bar=fig.colorbar(im,ax=heat,fraction=.035,pad=.025)
    bar.set_label("带符号的振幅 C；概率为 |C|²")
    fig.text(.68,.015,r"相位：每个 $j$ 的最高权态中，最大 $m_1$ 项取正；随后用 $J_-$ 生成。"+"\n"
             "数字显示到三位小数，精确系数见习题 3.34；颜色不表示混合概率。",
             ha="center",va="bottom",fontsize=10)
    for extension in ["png","svg"]:
        fig.savefig(OUT/f"3.8-耦合格点与CG矩阵.{extension}",dpi=190,bbox_inches="tight")
    plt.close(fig)


def main():
    cases=check_coupling()
    examples=sphere_checks()
    plot()
    pairs,labels,c,*_=couple(1.,1.)
    result={"status":"PASS","all_checks_passed":True,"section":"3.8 角动量的叠加",
            "source":"PDF223—236上部，印刷201—214上部；式3.321—3.393",
            "scope":"Teaching supplement; independent matrix construction and sphere quadrature, not evidence of learner mastery.",
            "hbar_units":1,"coupled_pairs":cases,"checks":CHECKS,
            "gaunt_examples":examples,"CG_matrix_1x1":{"rows_m1_m2":pairs,"columns_j_M":labels,"values":c.tolist()},
            "phase":"Maximum m1 coefficient of each highest-weight state positive, lower states from Jminus.",
            "quadrature":{"Gauss_Legendre_cos_theta_nodes":64,"uniform_phi_nodes":128}}
    (OUT/"3.8-角动量耦合核验结果.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"checks":CHECKS,"gaunt_examples":examples},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
