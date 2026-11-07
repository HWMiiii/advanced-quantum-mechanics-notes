"""Sakurai 中文第3版 §3.5：独立矩阵复算与教学配图。

依赖 NumPy、Matplotlib。执行后将两张图（PNG/SVG）与检查 JSON 保存到本目录。
检查中采用 ℏ=1；3.46 保留本项目扫描本的 2s(s-1)ℏ² 分母。
所有曲线由解析模型计算，不是实验数据，也不表示学习者已独立完成原题。
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261023)
FONT = Path("C:/Windows/Fonts/msyh.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
                     "axes.unicode_minus": False, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#f8fafc", "axes.facecolor": "white"})
BLUE, ORANGE, GREEN = "#2867a8", "#cc6b31", "#238577"
CHECKS = {}

def record(key, value, tol=1e-11):
    value = float(np.max(np.abs(value)))
    previous = CHECKS.get(key, {}).get("max_abs_error", 0.)
    value = max(value, previous)
    CHECKS[key] = {"max_abs_error": value, "tolerance": tol, "passed": value <= tol}
    assert value <= tol, (key, value, tol)

def angular(j):
    assert j >= 0 and float(2*j).is_integer()
    n = int(2*j+1)
    m = np.arange(j, -j-0.1, -1)
    jp = np.zeros((n,n), complex)
    for col in range(1,n):
        jp[col-1,col] = np.sqrt((j-m[col])*(j+m[col]+1))
    jm = jp.conj().T
    return m, (jp+jm)/2, (jp-jm)/(2j), np.diag(m), jp, jm

def rotation(generator, angle):
    val, vec = np.linalg.eigh(generator)
    return (vec*np.exp(-1j*angle*val)) @ vec.conj().T

def d_one(beta):
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[(1+c)/2,-s/np.sqrt(2),(1-c)/2],
                     [s/np.sqrt(2),c,-s/np.sqrt(2)],
                     [(1-c)/2,s/np.sqrt(2),(1+c)/2]])

def j2_column_zero(beta):
    c, s = np.cos(beta), np.sin(beta)
    return np.array([np.sqrt(6)/4*s*s,-np.sqrt(6)/2*s*c,
                     (3*c*c-1)/2,np.sqrt(6)/2*s*c,np.sqrt(6)/4*s*s])

def checks():
    js = [0,0.5,1,1.5,2,2.5,3,5]
    for j in js:
        m, x,y,z,jp,jm = angular(j)
        n, ident = len(m), np.eye(len(m))
        record("commutators", [np.max(abs(x@y-y@x-1j*z)),
                               np.max(abs(y@z-z@y-1j*x)),
                               np.max(abs(z@x-x@z-1j*y))])
        record("casimir", x@x+y@y+z@z-j*(j+1)*ident)
        record("ladder_products", [np.max(abs(jm@jp-(j*(j+1)*ident-z@z-z))),
                                    np.max(abs(jp@jm-(j*(j+1)*ident-z@z+z)))])
        record("component_spectra", [np.max(abs(np.linalg.eigvalsh(a)-m[::-1])) for a in [x,y,z]])
        record("ladder_endpoints", [np.linalg.norm(jp[:,0]), np.linalg.norm(jm[:,-1])])
        record("transverse_second_moments", np.diag(y@y)-(j*(j+1)-m*m)/2)
        record("two_pi_rotation", rotation(y,2*np.pi)-(-1)**int(2*j)*ident)
        for _ in range(20):
            alpha,beta,gamma = RNG.uniform(-np.pi,np.pi,3)
            d = rotation(y,beta)
            full = rotation(z,alpha) @ d @ rotation(z,gamma)
            factored = np.exp(-1j*m*alpha)[:,None]*d*np.exp(-1j*m*gamma)[None,:]
            record("euler_phase_indices", full-factored)
            record("rotation_unitarity", full.conj().T@full-ident)
            record("rotation_covariance", d.conj().T@z@d-(np.cos(beta)*z-np.sin(beta)*x))
            prob = abs(d)**2
            record("probability_normalization", prob.sum(axis=0)-1)
            # 原题3.37(a) 固定列m'；(b) 固定行m'，严格保留扫描指标次序。
            record("exercise_3_37a_column_first_moment", m@prob-m*np.cos(beta))
            target = j*(j+1)/2*np.sin(beta)**2+m*m/2*(3*np.cos(beta)**2-1)
            record("exercise_3_37b_row_second_moment", prob@(m*m)-target)
            a = RNG.normal(size=(n,n))+1j*RNG.normal(size=(n,n))
            rho = a@a.conj().T
            rho /= np.trace(rho)
            rotated = full@rho@full.conj().T
            record("density_rotation_purity", np.trace(rotated@rotated)-np.trace(rho@rho))
        eps = 2e-4
        prob_stay = abs(rotation(y,eps)[0,0])**2
        record("exercise_3_17_second_order", prob_stay-(1-j/2*eps**2), 1e-12)

    m,x,y,z,jp,jm = angular(1)
    for a in [x,y,z]:
        record("exercise_3_6_cubic_zero", a@a@a-a)
    vecx = np.array([[1,1,1],[np.sqrt(2),0,-np.sqrt(2)],[1,-1,1]],complex)
    vecx /= np.array([2,np.sqrt(2),2])
    vecy = np.array([[1,1,1],[1j*np.sqrt(2),0,-1j*np.sqrt(2)],[-1,1,-1]],complex)
    vecy /= np.array([2,np.sqrt(2),2])
    for op, vec in [(x,vecx),(y,vecy)]:
        record("exercise_3_20_eigenvectors", op@vec-vec*np.array([1,0,-1]))
        record("exercise_3_20_orthonormality", vec.conj().T@vec-np.eye(3))
    for beta in np.linspace(-2*np.pi,2*np.pi,129):
        exact = rotation(y,beta)
        poly = np.eye(3)-1j*y*np.sin(beta)-y@y*(1-np.cos(beta))
        record("exercise_3_38_polynomial", exact-poly)
        record("exercise_3_38_full_d1", exact-d_one(beta))
    m,x,y,z,jp,jm = angular(2)
    for beta in np.linspace(0,np.pi,121):
        amplitudes = rotation(y,beta)[:,2]
        predicted = j2_column_zero(beta)
        record("exercise_3_26_five_amplitudes", amplitudes-predicted)
        record("exercise_3_26_five_probabilities", np.sum(predicted**2)-1)

    m,x,y,z,sp,sm = angular(1.5)
    s2 = x@x+y@y+z@z
    boundary = [(1,0),(-1,0),(0,1),(0,-1),(0,0)]
    samples = boundary+list(map(tuple,RNG.normal(size=(100,2))))
    for a,b in samples:
        ham = a*(3*z@z-s2)+b*(sp@sp+sm@sm)
        r = np.sqrt(9*a*a+12*b*b)
        record("exercise_3_46_energy_spectrum", np.linalg.eigvalsh(ham)-np.array([-r,-r,r,r]))
        if r == 0:
            record("exercise_3_46_zero_boundary", ham)
            continue
        theta = np.arctan2(2*np.sqrt(3)*b,3*a)
        c, st = np.cos(theta/2),np.sin(theta/2)
        eigvecs = np.zeros((4,4),complex)
        # 正能、负能在两同构块的固定基顺序。
        eigvecs[[0,2],0] = [c,st]
        eigvecs[[3,1],1] = [c,st]
        eigvecs[[0,2],2] = [-st,c]
        eigvecs[[3,1],3] = [-st,c]
        record("exercise_3_46_all_eigenvectors", ham@eigvecs-eigvecs*np.array([r,r,-r,-r]))
        record("exercise_3_46_orthonormality", eigvecs.conj().T@eigvecs-np.eye(4))
    # 原题的实际前因子检查，独立于抽象A、B谱检查。ℏ=eQ=1。
    for _ in range(100):
        vx,vy = RNG.normal(size=2)
        vz = -vx-vy
        s = 1.5
        prefactor = 1/(2*s*(s-1))
        a,b = prefactor*vz/2, prefactor*(vx-vy)/4
        original = prefactor*(vx*x@x+vy*y@y+vz*z@z)
        rewritten = a*(3*z@z-s2)+b*(sp@sp+sm@sm)
        record("exercise_3_46_scanned_prefactor_rewriting", original-rewritten)
        magnitude = np.sqrt(vz*vz+(vx-vy)**2/3)
        record("exercise_3_46_potential_derivative_spectrum",
               np.linalg.eigvalsh(original)-np.array([-magnitude,-magnitude,magnitude,magnitude]))
    return {"j_values":js, "random_euler_tests_per_j":20,
            "exercise_3_26_beta_samples":121, "exercise_3_46_A_B_samples":len(samples),
            "exercise_3_46_scanned_prefactor":"eQ / [2s(s-1) hbar^2]",
            "j1_beta_pi_over_2_probabilities":[0.25,0.5,0.25],
            "j2_beta_pi_over_2_probabilities_by_m_2_1_0_minus1_minus2":[0.375,0,0.25,0,0.375]}

def style(ax):
    ax.grid(alpha=0.16)
    ax.set_axisbelow(True)

def save(fig,name):
    fig.savefig(OUT/(name+".png"),dpi=180,bbox_inches="tight")
    fig.savefig(OUT/(name+".svg"),bbox_inches="tight")
    plt.close(fig)

def figures():
    fig,axes = plt.subplots(1,2,figsize=(13,5.1),layout="constrained")
    ax = axes[0]
    for pos,j in enumerate([0,0.5,1,1.5]):
        m = np.arange(-j,j+0.1,1)
        for value in m:
            ax.plot([pos-.23,pos+.23],[value,value],color=BLUE,lw=2.8)
            ax.scatter([pos],[value],s=30,color=BLUE,zorder=4)
        if j:
            ax.annotate("",xy=(pos+.3,j),xytext=(pos+.3,-j),
                        arrowprops={"arrowstyle":"->","color":ORANGE,"lw":1.7})
            ax.text(pos+.33,0,r"$J_+$",color=ORANGE,va="center",fontsize=11)
        ax.text(pos,1.83,str(int(2*j+1))+" 个态",ha="center",fontsize=11)
    ax.set_xticks(range(4),[r"$j=0$",r"$j=1/2$",r"$j=1$",r"$j=3/2$"])
    ax.set(title="固定 j：向上、向下都必须终止",ylabel=r"$m=J_z/\hbar$（不是能量）",ylim=(-1.85,2.05),xlim=(-.6,3.7))
    ax.set_yticks(np.arange(-1.5,1.51,.5))
    style(ax)
    ax = axes[1]
    j = np.arange(0,4.01,0.5)
    ax.plot(j,np.sqrt(j*(j+1)),color=BLUE,lw=2.5,marker="o",label=r"大小 $\sqrt{j(j+1)}$")
    ax.plot(j,j,color=ORANGE,lw=2.5,ls="--",marker="s",label=r"分量最大值 $j$")
    ax.plot(j,np.sqrt(j/2),color=GREEN,lw=2.3,ls="-.",marker="^",label=r"最高权态 $\Delta J_x/\hbar=\sqrt{j/2}$")
    ax.set(title="最高权态仍有横向量子涨落\n（j 以 1/2 为步长，连线仅作导引）",xlabel=r"$j$",ylabel=r"以 $\hbar$ 为单位")
    ax.legend(loc="upper left",fontsize=10)
    style(ax)
    fig.suptitle("§3.5  角动量的有限阶梯与横向涨落",fontsize=16)
    save(fig,"3.5-角动量阶梯与横向涨落")

    fig,axes = plt.subplots(1,2,figsize=(13,4.8),layout="constrained")
    beta = np.linspace(0,np.pi,601)
    ax = axes[0]
    ax.plot(beta,np.cos(beta/2)**4,color=BLUE,lw=2.5,label=r"$m'=+1$")
    ax.plot(beta,np.sin(beta)**2/2,color=GREEN,lw=2.5,label=r"$m'=0$")
    ax.plot(beta,np.sin(beta/2)**4,color=ORANGE,lw=2.5,label=r"$m'=-1$")
    ax.set_title(r"$j=1,\ m=1$：读取 $d^{(1)}$ 的第一列")
    ax.legend(loc="upper center",fontsize=10)
    ax = axes[1]
    c,s = np.cos(beta),np.sin(beta)
    ax.plot(beta,(3*c*c-1)**2/4,color=BLUE,lw=2.5,label=r"$m'=0$")
    ax.plot(beta,1.5*s*s*c*c,color=ORANGE,lw=2.5,label=r"$m'=\pm1$（每个态）")
    ax.plot(beta,3*s**4/8,color=GREEN,lw=2.5,label=r"$m'=\pm2$（每个态）")
    ax.set_title(r"原题 3.26：$j=2,\ m=0$ 的五种结果")
    ax.legend(loc="upper center",fontsize=10)
    for ax in axes:
        ax.set(xlabel=r"绕 y 轴的旋转角 $\beta$",ylabel="测得指定 m' 的概率",ylim=(-.035,1.05))
        ax.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi],
                      [r"$0$",r"$\pi/4$",r"$\pi/2$",r"$3\pi/4$",r"$\pi$"])
        style(ax)
    fig.suptitle("§3.5  同一旋转，不同角动量态的测量分布",fontsize=16)
    save(fig,"3.5-自旋旋转概率")

if __name__ == "__main__":
    details = checks()
    figures()
    result = {"status":"PASS","section":"3.5","seed":20261023,"hbar_in_checks":1,
              "checks":CHECKS,"examples":details,
              "meaning":"仅为解析公式的独立数值核验；不替代证明或学习者的独立作答。"}
    path = OUT/"3.5-角动量与旋转核验结果.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"PASS","check_groups":len(CHECKS),"figures":2,
                      "maximum_residual":max(v["max_abs_error"] for v in CHECKS.values()),
                      "result":str(path)},ensure_ascii=False))
