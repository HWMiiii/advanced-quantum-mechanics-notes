"""§4.4 教学补充的有限矩阵核验与科学图；不改动教材原 PDF。"""
from pathlib import Path
import os
import sys
import json
import math
import csv

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
sys.path.insert(0, str(ROOT / "tmp/section3_1_deps"))
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp/pdfs/section4_4/mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 11,
                     "axes.unicode_minus": False, "svg.fonttype": "none"})
checks = []

def check(name, residual, tolerance=1e-11):
    value = float(np.max(np.abs(residual)))
    checks.append({"name": name, "residual": value,
                   "tolerance": tolerance, "passed": value < tolerance})

def spin(j):
    m = np.arange(j, -j-1, -1)
    jp = np.zeros((len(m), len(m)), dtype=complex)
    u = np.zeros_like(jp)
    for col, mm in enumerate(m):
        if col > 0:
            jp[col-1, col] = np.sqrt(j*(j+1)-mm*(mm+1))
        u[len(m)-1-col, col] = (-1)**int(round(j-mm))
    return (jp+jp.conj().T)/2, (jp-jp.conj().T)/(2j), np.diag(m), u

def normalize(v):
    return v / np.linalg.norm(v)

def exp_hermitian(h, t):
    e, v = np.linalg.eigh(h)
    return (v*np.exp(-1j*e*t)) @ v.conj().T

sx, sy, sz, uhalf = spin(.5)
sigmas = [2*sx, 2*sy, 2*sz]
rng = np.random.default_rng(44194)
psi = normalize(rng.normal(size=2)+1j*rng.normal(size=2))
phi = normalize(rng.normal(size=2)+1j*rng.normal(size=2))
theta = lambda v: uhalf @ v.conj()
check("自旋1/2：Theta²=-I", uhalf @ uhalf.conj() + np.eye(2))
check("反幺正内积次序", np.vdot(theta(phi), theta(psi))-np.vdot(psi, phi))
check("任意旋量与时间反演伙伴正交", np.vdot(psi, theta(psi)))
check("三分量自旋都翻转", [np.vdot(theta(psi), s @ theta(psi)) + np.vdot(psi, s @ psi) for s in sigmas])
x = rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
theta_xdag = uhalf @ x.conj().T.conj() @ uhalf.conj().T
check("教材4.139一般非厄米X矩阵元", np.vdot(phi, x @ psi)-np.vdot(theta(psi), theta_xdag @ theta(phi)))

w, _ = np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
uprime = w.conj().T @ uhalf @ w.conj()
cprime = w.conj().T @ psi
check("反幺正基底变换U'=W†UW*", uprime @ cprime.conj()-w.conj().T @ theta(psi))
check("变换基底后Theta²仍=-I", uprime @ uprime.conj()+np.eye(2))
old_k_new = w.conj().T @ w.conj()
check("原K在新基中的矩阵W†W*", old_k_new @ cprime.conj()-w.conj().T @ psi.conj())
eta = np.exp(1j*.731)
check("Theta整体相位不改变平方", (eta*uhalf) @ (eta*uhalf).conj()+np.eye(2))

jx, jy, jz, uthreehalf = spin(1.5)
ident4 = np.eye(4)
D, E = 1.0, .35
nhat = np.array([1.,2.,3.])/np.sqrt(14)
hq = D*(jz @ jz-1.25*ident4)+E*(jx @ jx-jy @ jy)
zeeman = nhat[0]*jx+nhat[1]*jy+nhat[2]*jz
h = lambda b: hq+b*zeeman
tra = lambda a: uthreehalf @ a.conj() @ uthreehalf.conj().T
energies0, vectors0 = np.linalg.eigh(hq)
rho = np.sqrt(D*D+3*E*E)
check("j=3/2：Theta²=-I", uthreehalf @ uthreehalf.conj()+ident4)
check("电四极哈密顿量厄米", hq-hq.conj().T)
check("电四极哈密顿量时间反演不变", tra(hq)-hq)
check("零场双重谱解析式", energies0-np.array([-rho,-rho,rho,rho]))
for col in range(4):
    v = vectors0[:,col]
    tv = uthreehalf @ v.conj()
    check(f"零场本征态{col+1}伙伴同能", hq @ tv-energies0[col]*tv)
    check(f"零场本征态{col+1}伙伴正交", np.vdot(v,tv))
b_fixed = .4
energies_fixed = np.linalg.eigvalsh(h(b_fixed))
check("外场反转的算符关系ThetaH(b)Theta^-1=H(-b)", tra(h(b_fixed))-h(-b_fixed))
check("正反磁场谱集合相同", energies_fixed-np.linalg.eigvalsh(h(-b_fixed)))
check("固定磁场对称性破坏量满足解析式", tra(h(b_fixed))-h(b_fixed)+2*b_fixed*zeeman)
check("电四极与Zeeman哈密顿量迹均为零", [np.trace(hq),np.trace(h(b_fixed))])
time = .71
evolution0 = exp_hermitian(hq,time)
check("Theta U(t) Theta^-1=U(-t)", tra(evolution0)-exp_hermitian(hq,-time))

# 4.12: choose the exact textbook i^(2m) convention, not an inferred phase.
A, B = 1.4, .35
hspin1 = np.array([[A,0,B],[0,0,0],[B,0,A]],dtype=complex)
uone = np.array([[0,0,-1],[0,1,0],[-1,0,0]],dtype=complex)
vone = np.column_stack(([0,1,0],np.array([1,0,-1])/np.sqrt(2),np.array([1,0,1])/np.sqrt(2)))
eone = np.array([0,A-B,A+B])
theta_phases = np.array([1,1,-1])
check("4.12三个态归一正交", vone.conj().T @ vone-np.eye(3))
check("4.12三本征方程", hspin1 @ vone-vone*eone)
check("4.12自旋1时间反演不变", uone @ hspin1.conj() @ uone.conj().T-hspin1)
check("4.12自旋1的Theta²=I", uone @ uone.conj()-np.eye(3))
check("4.12教材相位下三个Theta相位", uone @ vone.conj()-vone*theta_phases)

# Counterexample to using 4.183 without a fixed alpha multiplet closed under Theta.
# The order is s+,s-,p+,p-, with real radial wavefunctions and real CG coefficients.
radial_exact = (2*math.factorial(4)-math.factorial(5))/(8*np.sqrt(3))
q, weight = np.polynomial.legendre.leggauss(160)
r = (q+1)*40
radial_num = np.sum(weight*40*r**4*(2-r)*np.exp(-r)/(8*np.sqrt(3)))
check("氢n=2径向积分数值与解析一致", radial_num-radial_exact)
d = -radial_exact/3
taux, tauy, tauz = sigmas
usp = np.kron(tauz,uhalf)
coordinates = [d*np.kron(taux,ss) for ss in sigmas]
rotations = [np.kron(np.eye(2),ss/2) for ss in sigmas]
mix = normalize(np.array([1,0,1,0],dtype=complex))
partner = usp @ mix.conj()
mix_samealpha_minus = normalize(np.array([0,1,0,1],dtype=complex))
check("s/p时间反演平方=-I", usp @ usp.conj()+ident4)
check("投影位置向量在时间反演下为偶", [usp @ xx.conj() @ usp.conj().T-xx for xx in coordinates])
check("投影位置向量为旋转矢量xy分量", rotations[0] @ coordinates[1]-coordinates[1] @ rotations[0]-1j*coordinates[2])
check("实s/p等权混合的非零z期望", np.vdot(mix,coordinates[2] @ mix)-np.sqrt(3))
check("时间反演伙伴保持位置期望", np.vdot(partner,coordinates[2] @ partner)-np.vdot(mix,coordinates[2] @ mix))
check("该伙伴不属于原alpha的m负态", np.vdot(mix_samealpha_minus,partner))
closed_mix = normalize(np.array([1,0,1j,0],dtype=complex))
closed_minus = normalize(np.array([0,1,0,1j],dtype=complex))
check("纯虚相对系数的alpha多重态闭合", usp @ closed_mix.conj()-closed_minus)
check("闭合多重态的z期望为零", np.vdot(closed_mix,coordinates[2] @ closed_mix))

field_grid = np.linspace(-.8,.8,321)
spectrum = np.array([np.linalg.eigvalsh(h(bb)) for bb in field_grid])
# Keep the human-facing data filename free of slashes from j=3/2.
csvpath = OUT/"4.4-自旋3_2克拉默斯谱.csv"
with csvpath.open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["b_over_D","energy_1_over_D","energy_2_over_D","energy_3_over_D","energy_4_over_D"])
    writer.writerows(np.column_stack((field_grid/D,spectrum/D)))

fig, axs = plt.subplots(1,2,figsize=(11.5,5.8),gridspec_kw={"width_ratios":[1,2.1]},layout="constrained")
colors=["#26547c","#168c9c","#e09b24","#c04c56"]
ax=axs[0]
for xx,energies in [(0,energies0),(1,energies_fixed)]:
    for idx, en in enumerate(energies):
        # Draw two coincident zero-field states on adjoining horizontal segments.
        if xx==0:
            left,right=(-.28,0) if idx%2==0 else (0,.28)
        else:
            left,right=(.76,1.24)
        ax.plot([left,right],[en/D,en/D],lw=3.5,color=colors[idx])
for yy in [-rho,rho]:
    ax.annotate("二重",(.0,yy/D),xytext=(.02,yy/D+.15),fontsize=10)
ax.set(xticks=[0,1],xticklabels=["零磁场\n$b=0$","固定磁场\n$b=0.4D$"],ylabel="$E/D$",xlim=(-.5,1.55),ylim=(-1.9,1.9),title="同一哈密顿量内的简并是否存在")
ax.grid(axis="y",alpha=.2)
ax=axs[1]
for idx in range(4):
    ax.plot(field_grid/D,spectrum[:,idx]/D,lw=2.2,color=colors[idx])
ax.axvline(0,color="#555555",lw=.9,ls="--")
ax.axvline(.4,color="#777777",lw=.9,ls=":")
ax.scatter([0,0],[-rho/D,rho/D],s=65,facecolors="white",edgecolors="#222222",zorder=5)
ax.text(.03,0,"零场：两对克拉默斯伙伴\n非零场：本例中四能级分离",fontsize=10,bbox={"boxstyle":"round,pad=.4","fc":"white","ec":"#dddddd"})
ax.set(xlabel="固定外磁场的耦合系数 $b/D$",ylabel="$E/D$",title="反转磁场后谱集合相同，不代表固定磁场保持时间反演")
ax.grid(alpha=.2)
fig.suptitle("自旋 $j=3/2$：电四极各向异性保留双重简并，磁场使其劈裂",fontsize=15)
fig.supxlabel(r"$H_Q=D(j_z^2-5/4)+0.35D(j_x^2-j_y^2)$，$H(b)=H_Q+b\,(j_x+2j_y+3j_z)/\sqrt{14}$",fontsize=11)
for ext in ["png","svg"]:
    fig.savefig(OUT/f"4.4-克拉默斯简并与磁场劈裂.{ext}",dpi=180,bbox_inches="tight")
plt.close(fig)

report={"date":"2026-10-30","purpose":"§4.4教学补充；独立数值核验，不表示学习完成",
        "all_passed":all(c["passed"] for c in checks),"number_of_checks":len(checks),
        "maximum_residual":max(c["residual"] for c in checks),"checks":checks,
        "quadrupole":{"j":1.5,"D":D,"E":E,"field_direction":nhat.tolist(),"zero_field_energies":energies0.tolist(),"b_fixed":b_fixed,"fixed_field_energies":energies_fixed.tolist(),"minimum_fixed_field_gap":float(np.min(np.diff(energies_fixed))),"fixed_field_time_reversal_breaking_norm":float(np.linalg.norm(tra(h(b_fixed))-h(b_fixed)))},
        "spin1_exercise4_12":{"A":A,"B":B,"hbar":1,"eigenvalues":eone.tolist(),"theta_phases":theta_phases.tolist()},
        "s_p_counterexample":{"hydrogen_n2_radial_integral_in_a0":float(radial_exact),"numerical_radial_integral_in_a0":float(radial_num),"equal_real_mixture_z_in_a0":float(d),"same_alpha_closure":"In the stated real radial/CG basis, cp/cs must be purely imaginary when both coefficients are nonzero."},
        "assets":[csvpath.name,"4.4-克拉默斯简并与磁场劈裂.png","4.4-克拉默斯简并与磁场劈裂.svg"]}
(OUT/"4.4-反幺正与克拉默斯核验结果.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:report[k] for k in ["all_passed","number_of_checks","maximum_residual","quadrupole","spin1_exercise4_12","s_p_counterexample"]},ensure_ascii=False,indent=2))
assert report["all_passed"],"存在未通过的数值核验"
assert np.min(np.diff(energies_fixed)) > 1e-3, "指定非零磁场仍有简并，需重新检查"
