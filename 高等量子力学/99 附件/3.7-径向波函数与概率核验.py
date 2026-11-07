"""Sakurai 3.7 supplementary radial figures and analytic checks.
Requires numpy and matplotlib. All checks use hbar=M=e^2=Z=1.
Mandatory finite-well computation is in a separate file.
"""
from pathlib import Path
import json, math
import numpy as np
from numpy.polynomial import Polynomial as Poly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
plt.rcParams.update({"font.family":["Microsoft YaHei","SimHei","DejaVu Sans"],
    "mathtext.fontset":"dejavusans","axes.unicode_minus":False,
    "font.size":12,"axes.titlesize":14,"axes.labelsize":12,
    "figure.facecolor":"#f7f9fc","axes.facecolor":"white",
    "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
COLORS=["#2b6cb0","#d77b31","#208778"]
def laguerre_poly(s,alpha):
    return Poly([(-1)**k*math.comb(s+alpha,s-k)/math.factorial(k) for k in range(s+1)])
def radial_poly(n,l):
    s=n-l-1
    x=Poly([0,2/n])
    p=(x**l)*laguerre_poly(s,2*l+1)(x)
    c=(2/n)**1.5*math.sqrt(math.factorial(s)/(2*n*math.factorial(n+l)))
    return c*p
def radial(n,l,r):
    return np.exp(-np.asarray(r)/n)*radial_poly(n,l)(r)
def savefig(fig,name):
    fig.savefig(OUT/(name+".png"),dpi=180,bbox_inches="tight")
    fig.savefig(OUT/(name+".svg"),bbox_inches="tight")
    plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(13.2,5.6))
x=np.linspace(.055,8,2000)
for l,c in zip(range(3),COLORS):
    axs[0].plot(x,-1/x+l*(l+1)/(2*x*x),color=c,lw=2.4,label=f"$l={l}$")
axs[0].set(xlim=(0,8),ylim=(-4.5,3.5),xlabel=r"$r/a_0$",
    ylabel=r"$V_{\rm eff}/(\hbar^2/Ma_0^2)$",
    title="① 库仑有效势：角动量带来的离心势垒")
axs[0].axhline(0,color="#9daabd",lw=.8)
axs[0].legend(frameon=False,loc="upper right")
s=np.geomspace(.008,.4,250)
for l,c in zip(range(3),COLORS):
    axs[1].loglog(s,s**(2*l+3)/(2*l+3),color=c,lw=2.4,label=f"l = {l}：幂次 {2*l+3}")
axs[1].set(xlabel=r"小球半径 $s=\varepsilon/a_0$",
    ylabel=r"局部积分 $\int_0^s t^{2l+2}dt$",
    title="② 正规支的小球概率：包含球壳面积因子")
axs[1].legend(frameon=False,loc="upper left")
for ax in axs: ax.grid(alpha=.22)
fig.suptitle(r"原点边界决定正规支：$u\sim r^{l+1}$，$R\sim r^l$",fontsize=19,y=1.02)
fig.subplots_adjust(bottom=.22,wspace=.3,top=.84)
fig.text(.06,.04,
    "② 以相同局部系数比较幂律，仅示原点渐近；不是三条完整、归一化的原子概率分布。\n"
    "绝对小球概率 ∝ ε^(2l+3)；相对 s 态的额外抑制才含 ε^(2l)。",fontsize=11)
savefig(fig,"3.7-有效势与原点边界")

r=np.linspace(0,15,1800)
fig,axs=plt.subplots(1,2,figsize=(13.2,5.6))
for (n,l,label),c in zip([(1,0,"1s"),(2,0,"2s"),(2,1,"2p")],COLORS):
    R=radial(n,l,r)
    axs[0].plot(r,R,lw=2.4,color=c,label=label)
    axs[1].plot(r,r*r*R*R,lw=2.4,color=c,label=label)
axs[0].set(xlim=(0,12),ylim=(-.13,2.06),xlabel=r"$r/a_Z$",
    ylabel=r"$a_Z^{3/2} R_{nl}(r)$",title="① 径向函数 R：可有正负号与节点")
axs[1].set(xlim=(0,15),ylim=(0,.58),xlabel=r"$r/a_Z$",
    ylabel=r"$a_Z r^2 |R_{nl}(r)|^2$",title="② 径向概率密度：全半轴积分均为 1")
axs[0].scatter([2],[0],color=COLORS[1],s=30,zorder=5)
axs[0].annotate(r"2s 节点：$r=2a_Z$",xy=(2,0),xytext=(4.1,.48),
    arrowprops={"arrowstyle":"->","color":COLORS[1]},color=COLORS[1])
for ax in axs:
    ax.axhline(0,color="#a8b4c4",lw=.7)
    ax.legend(frameon=False)
    ax.grid(alpha=.22)
fig.suptitle("氢样原子的径向函数与径向测量",fontsize=19,y=1.02)
fig.subplots_adjust(bottom=.2,wspace=.28,top=.84)
fig.text(.06,.04,
    "a_Z = a_0/Z。球谐函数已归一，因此 dP = r²|R|²dr = |u|²dr，不再额外乘 4π。\n"
    "1s 的 R 在原点最大，径向概率密度却在 r = a_Z 最大；图中不含自旋。",fontsize=11)
savefig(fig,"3.7-氢样原子径向函数与概率")

z,w=np.polynomial.laguerre.laggauss(80)
checks={}
def check(name,value,tol=2e-10):
    checks[name]={"max_abs_error":float(value),"tolerance":tol,"passed":bool(value<tol)}
normerr=ortherr=odeerr=0.
for n in range(1,7):
    for l in range(n):
        p=radial_poly(n,l)
        rr=z*n/2
        norm=np.sum(w*rr**2*p(rr)**2)*(n/2)
        normerr=max(normerr,abs(norm-1))
        grid=np.geomspace(.04,35,400)
        exp=np.exp(-grid/n)
        R=exp*p(grid)
        Rp=exp*(p.deriv()(grid)-p(grid)/n)
        Rpp=exp*(p.deriv(2)(grid)-2*p.deriv()(grid)/n+p(grid)/n**2)
        u=grid*R
        u2=2*Rp+grid*Rpp
        residual=-.5*u2+(l*(l+1)/(2*grid**2)-1/grid)*u+u/(2*n*n)
        odeerr=max(odeerr,np.max(np.abs(residual)))
        for nn in range(l+1,n):
            alpha=1/n+1/nn
            rr=z/alpha
            overlap=np.sum(w*rr**2*p(rr)*radial_poly(nn,l)(rr))/alpha
            ortherr=max(ortherr,abs(overlap))
check("hydrogen_radial_normalization_n_1_to_6",normerr)
check("same_l_different_n_orthogonality",ortherr)
check("radial_schrodinger_equation_hbar_M_g_1",odeerr)
means={}
for n,l,label,expected in [(1,0,"1s",1.5),(2,0,"2s",6),(2,1,"2p",5)]:
    rr=z*n/2
    mean=np.sum(w*rr**3*radial_poly(n,l)(rr)**2)*(n/2)
    means[label]=float(mean)
    check("mean_radius_"+label,abs(mean-expected))
check("2s_radial_node",abs(radial(2,0,2)))
count_error=max(abs(sum(2*(N-2*q)+1 for q in range(N//2+1))-(N+1)*(N+2)//2)
                for N in range(31))
check("oscillator_degeneracy_N_0_to_30",count_error,1e-12)
result={"section":"3.7 中心势的薛定谔方程",
    "purpose":"教学补充；原题3.28的指定数值另存球形势阱核验文件",
    "units":"hbar=M=e^2=Z=1; a_0=1","checks":checks,
    "all_checks_passed":all(v["passed"] for v in checks.values()),
    "mean_r_over_aZ":means,
    "outputs":["3.7-有效势与原点边界.png","3.7-有效势与原点边界.svg",
               "3.7-氢样原子径向函数与概率.png","3.7-氢样原子径向函数与概率.svg"]}
(OUT/"3.7-径向波函数与概率核验结果.json").write_text(
    json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))
assert result["all_checks_passed"]

