"""§5.1 教学计算：两态、改变刚度的振子、氢基态Stark效应。

全部参数扫描为教学补充，不冒充章末题要求。教材 PDF 不作任何写入。
依赖 numpy、matplotlib；当前项目可用 bundled Python 3.12 运行。
"""
from pathlib import Path
import os
import sys
import math
import json
import csv

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
LOCAL_DEPS = ROOT / "tmp/section3_1_deps"
if sys.version_info[:2] == (3, 12) and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp/section5_1_calculations/mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10.5,
                     "axes.unicode_minus": False, "svg.fonttype": "none",
                     "figure.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False})
checks = []

def check(name, residual, tolerance=1e-10):
    value = float(np.max(np.abs(residual)))
    checks.append({"name": name, "residual": value,
                   "tolerance": tolerance, "passed": bool(value < tolerance)})

def save_plot(fig, stem):
    fig.savefig(OUT / (stem + ".png"), dpi=190, bbox_inches="tight")
    fig.savefig(OUT / (stem + ".svg"), bbox_inches="tight")
    plt.close(fig)

def write_csv(name, columns, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)
        writer.writerows(rows)

# 1. Two-state system; lower unperturbed energy = 0, upper = 1.
two_samples = []
for g in [.03, .1, .3, .49, .7]:
    phase = .731
    off = g * np.exp(1j * phase)
    h = np.array([[0, off], [off.conjugate(), 1]], dtype=complex)
    energies, vectors = np.linalg.eigh(h)
    e = (1 - np.sqrt(1 + 4*g*g))/2
    z = .5*(1 + 1/np.sqrt(1 + 4*g*g))
    v = vectors[:, 0]
    v *= np.exp(-1j*np.angle(v[0]))
    check(f"两态 g={g} 精确能量", energies[0]-e)
    check(f"两态 g={g} 精确重叠Z", abs(v[0])**2-z)
    check(f"两态 g={g} 本征方程", h@v-e*v)
    c = v/v[0]
    check(f"两态 g={g} 中间归一化后恢复", c*np.sqrt(z)-v)
    delta = 1e-5
    lower_e = lambda x: .5*(x+1-np.sqrt((1-x)**2+4*g*g))
    derivative = (lower_e(delta)-lower_e(-delta))/(2*delta)
    check(f"两态 g={g} 能量对角元导数=Z", derivative-z, 1e-8)
    two_samples.append({"g_over_gap":g,"E_exact_over_gap":e,
                        "E_second_over_gap":-g*g,"Z_exact":z,
                        "Z_second":1-g*g})
gs=np.linspace(0,.8,401)
elow=(1-np.sqrt(1+4*gs**2))/2
ehigh=(1+np.sqrt(1+4*gs**2))/2
zexact=.5*(1+1/np.sqrt(1+4*gs**2))
write_csv("5.1-两态扫描.csv",
          ["coupling_over_gap","lower_exact","upper_exact","lower_second","upper_second","Z_exact","Z_second"],
          zip(gs,elow,ehigh,-gs**2,1+gs**2,zexact,1-gs**2))
fig,axs=plt.subplots(1,2,figsize=(12.5,4.8),layout="constrained")
axs[0].plot(gs,elow,color="#16697a",lw=2.2,label="精确下能级")
axs[0].plot(gs,ehigh,color="#c05424",lw=2.2,label="精确上能级")
axs[0].plot(gs,-gs**2,"--",color="#16697a",label="二阶下能级")
axs[0].plot(gs,1+gs**2,"--",color="#c05424",label="二阶上能级")
axs[0].set(ylabel="能量 / 原能隙",title="混合使两条能级互相排斥")
axs[0].legend(loc="center left",fontsize=9)
axs[1].plot(gs,zexact,lw=2.3,color="#16697a",label="精确 Z")
axs[1].plot(gs,1-gs**2,"--",color="#c05424",label="二阶近似 1 − η²")
axs[1].set(ylabel="原来低能态所占概率 Z",title="能量收敛与低阶精度是不同问题",ylim=(.3,1.03))
axs[1].legend(loc="lower left")
for ax in axs:
    ax.axvline(.5,ls=":",color="#595959",lw=1.5)
    ax.set(xlabel="无量纲混合强度 η = |λV₁₂| / 原能隙",xlim=(0,.8))
    ax.grid(alpha=.2)
fig.suptitle("两态模型：虚线竖线 η = 1/2 是展开的收敛圆边界；η ≪ 1 时低阶近似最可靠",fontsize=12.5)
save_plot(fig,"5.1-两态能级与混合概率")

# 2. Harmonic oscillator, hbar=m=omega=1, H=H0+epsilon W.
dim=48
a=np.zeros((dim,dim))
for n in range(1,dim):a[n-1,n]=np.sqrt(n)
x=(a+a.T)/np.sqrt(2)
h0=np.diag(np.arange(dim)+.5)
w=x@x/2
v0=np.eye(dim)[:,0]
v1=np.zeros(dim);v1[2]=-1/(4*np.sqrt(2))
v2=np.zeros(dim);v2[2]=1/(8*np.sqrt(2));v2[4]=np.sqrt(6)/64
e1=.25;e2=-1/16
check("振子一阶方程", (h0-.5*np.eye(dim))@v1+(w-e1*np.eye(dim))@v0)
check("振子二阶方程", (h0-.5*np.eye(dim))@v2+(w-e1*np.eye(dim))@v1-e2*v0)
check("振子一阶能量矩阵元", v0@w@v0-e1)
check("振子二阶能量矩阵元", v0@w@v1-e2)
v2_normalized=v2-v0/64
check("振子二阶归一化条件", 2*v0@v2_normalized+v1@v1)
osc_samples=[]
for eps in [.03,.1,.3,-.3]:
    es,vs=np.linalg.eigh(h0+eps*w)
    omega_new=np.sqrt(1+eps)
    z=2*np.sqrt(omega_new)/(1+omega_new)
    c0=np.sqrt(z)
    t=(1-omega_new)/(1+omega_new)
    coeff=np.zeros(dim)
    for k in range(dim//2):
        coeff[2*k]=c0*math.sqrt(math.factorial(2*k))/(2**k*math.factorial(k))*t**k
    numerical=vs[:,0]*np.sign(vs[0,0])
    check(f"振子 eps={eps} 精确基态能量", es[0]-.5*omega_new,1e-11)
    check(f"振子 eps={eps} 压缩态系数", numerical-coeff,1e-10)
    check(f"振子 eps={eps} 精确归一化", coeff@coeff-1,1e-11)
    osc_samples.append({"epsilon":eps,"E_exact_over_hw":.5*omega_new,
                        "E_second_over_hw":.5+eps/4-eps**2/16,
                        "Z_exact":z,"Z_second":1-eps**2/32,
                        "coefficient_0_exact":c0,"coefficient_2_exact":coeff[2],
                        "coefficient_4_exact":coeff[4],
                        "state_error_second":float(np.linalg.norm(numerical-(v0+eps*v1+eps**2*v2_normalized)))})
# Independent position-space Hermite quadrature for exact coefficients.
nodes,weights=np.polynomial.hermite.hermgauss(100)
eps=.3;s=np.sqrt(1+eps)
norm=(s**.25)/(np.sqrt(np.pi))
for n in [0,2,4]:
    cn=np.zeros(n+1);cn[n]=1
    hn=np.polynomial.hermite.hermval(nodes,cn)
    overlap=norm*np.sum(weights*hn*np.exp(-(s-1)*nodes**2/2))/np.sqrt(2**n*math.factorial(n))
    k=n//2
    exact_c=np.sqrt(2*np.sqrt(s)/(1+s))*math.sqrt(math.factorial(2*k))/(2**k*math.factorial(k))*((1-s)/(1+s))**k
    check(f"振子 n={n} 坐标积分交叉核验",overlap-exact_c)
epsgrid=np.linspace(-.7,.7,401)
exact_energy=.5*np.sqrt(1+epsgrid)
second_energy=.5+epsgrid/4-epsgrid**2/16
write_csv("5.1-振子能量扫描.csv",["epsilon","E_exact_over_hw","E_first_over_hw","E_second_over_hw"],
          zip(epsgrid,exact_energy,.5+epsgrid/4,second_energy))
xi=np.linspace(-3.5,3.5,501)
phi0=np.pi**(-.25)*np.exp(-xi**2/2)
phi2=(4*xi**2-2)*phi0/(2*np.sqrt(2))
phi4=(16*xi**4-48*xi**2+12)*phi0/(8*np.sqrt(6))
exact_wave=s**.25*np.pi**(-.25)*np.exp(-s*xi**2/2)
approx1=phi0-eps/(4*np.sqrt(2))*phi2
approx2=(1-eps**2/64)*phi0+(-eps/(4*np.sqrt(2))+eps**2/(8*np.sqrt(2)))*phi2+eps**2*np.sqrt(6)/64*phi4
write_csv("5.1-振子波函数比较.csv",["x_over_x0","psi_unperturbed","psi_exact","psi_first","psi_second"],
          zip(xi,phi0,exact_wave,approx1,approx2))
fig,axs=plt.subplots(1,2,figsize=(12.5,4.8),layout="constrained")
axs[0].plot(epsgrid,exact_energy,lw=2.2,color="#16697a",label="精确：√(1+ε)/2")
axs[0].plot(epsgrid,.5+epsgrid/4,ls=":",lw=2,color="#808080",label="一阶")
axs[0].plot(epsgrid,second_energy,ls="--",lw=2,color="#c05424",label="二阶")
axs[0].set(xlabel="刚度相对变化 ε",ylabel="基态能量 / ℏω",title="能量的二阶展开")
axs[1].plot(xi,phi0,lw=1.7,color="#808080",label="未微扰波函数")
axs[1].plot(xi,exact_wave,lw=2.3,color="#16697a",label="精确波函数")
axs[1].plot(xi,approx1,ls=":",color="#b07507",label="一阶波函数")
axs[1].plot(xi,approx2,ls="--",color="#c05424",label="二阶归一化展开")
axs[1].set(xlabel="位置 x / x₀",ylabel="波函数 × √x₀",title="ε = 0.30：刚度增加，波包变窄")
for ax in axs:ax.legend(fontsize=9);ax.grid(alpha=.2)
fig.suptitle("谐振子：只有偶数激发态混入；二阶态须补回沿原态的归一化项",fontsize=13)
save_plot(fig,"5.1-振子能量与波函数比较")

# 3. Hydrogen in atomic units, V=F z for electron charge e=-1.
# psi^(1)/psi0 = f(r) cos(theta), f=-r-r²/2.
rtest=np.linspace(.05,30,101)
f=-rtest-rtest**2/2
fp=-1-rtest
fpp=-np.ones_like(rtest)
ode=-.5*fpp+(1-1/rtest)*fp+f/rtest**2
check("氢基态一阶方程的径向恒等式",ode+rtest)
nodes,weights=np.polynomial.laguerre.laggauss(80)
rr=nodes/2
def average_r(values):return float(np.sum(weights*nodes**2/2*values))
r2=average_r(rr**2);r3=average_r(rr**3)
r4=average_r(rr**4)
check("氢基态 <r²>=3",r2-3,1e-9)
check("氢基态 <r³>=15/2",r3-7.5,1e-9)
e2_h=average_r(rr*(-rr-rr**2/2))/3
norm1=average_r((-rr-rr**2/2)**2)/3
check("氢基态二阶能量系数=-9/4",e2_h+9/4,1e-9)
check("氢基态一阶态范数=43/8",norm1-43/8,1e-9)
alpha=-2*e2_h
check("氢基态精确极化率=9/2",alpha-4.5,1e-9)
alpha_bound=2*1/(3/8)
check("闭合性极化率上界=16/3",alpha_bound-16/3)
fields=np.linspace(-.06,.06,301)
shift=-2.25*fields**2
write_csv("5.1-氢基态Stark扫描.csv",["field_atomic_unit","second_order_energy_shift_hartree","induced_dipole_atomic_unit","leakage_second"],
          zip(fields,shift,4.5*fields,43/8*fields**2))
linez=np.linspace(-5,5,501)
field=.03
rho0=np.exp(-2*np.abs(linez))/np.pi
fline=-(linez+.5*linez*np.abs(linez))
delta_rho=2*field*fline*rho0
write_csv("5.1-氢基态轴线密度响应.csv",["z_over_a0","rho0_times_a0_cubed","first_order_delta_rho_times_a0_cubed"],
          zip(linez,rho0,delta_rho))
fig,axs=plt.subplots(1,2,figsize=(12.5,4.8),layout="constrained")
axs[0].plot(fields,shift,color="#16697a",lw=2.3,label="二阶能移：−(9/4)F²")
axs[0].plot(fields,-.5*alpha_bound*fields**2,color="#c05424",ls="--",label="上界 α=16/3 对应的能移界")
axs[0].set(xlabel="外电场 F / 原子单位",ylabel="基态能移 / Hartree",title="极化率 4.5 小于上界 5.333…")
axs[0].legend(fontsize=9)
axs[1].plot(linez,delta_rho,color="#16697a",lw=2.3)
axs[1].fill_between(linez,0,delta_rho,where=delta_rho>0,color="#16697a",alpha=.2)
axs[1].fill_between(linez,0,delta_rho,where=delta_rho<0,color="#c05424",alpha=.2)
axs[1].axhline(0,color="#777777",lw=.8)
axs[1].set(xlabel="电场轴上的位置 z / a₀",ylabel="一阶密度变化 × a₀³",title="F = +0.03：电子密度向负 z 侧偏移")
for ax in axs:ax.grid(alpha=.2)
fig.suptitle("氢基态二次 Stark 效应：能量为场强偶函数，诱导电偶极矩沿外场方向",fontsize=13)
save_plot(fig,"5.1-氢基态二次Stark效应")

summary={"description":"教学核验；不计作教材章末原题或用户学习完成",
         "source_pages":"PDF306—307、312—316／印刷284—285、290—294；逐页查看扫描",
         "two_state_samples":two_samples,"oscillator_samples":osc_samples,
         "hydrogen_atomic_units":{"r2":r2,"r3":r3,"r4":r4,
         "E2_coefficient":e2_h,"first_order_state_norm_squared":norm1,
         "polarizability":alpha,"polarizability_upper_bound":alpha_bound},
         "checks":checks,"passed":all(c["passed"] for c in checks),
         "check_count":len(checks),"maximum_residual":max(c["residual"] for c in checks)}
(OUT/"5.1-非简并微扰核验结果.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"passed":summary["passed"],"check_count":len(checks),
                  "maximum_residual":summary["maximum_residual"]},ensure_ascii=False))
if not summary["passed"]:raise SystemExit("A numerical check failed")
