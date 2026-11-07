"""§5.8教学核验：K壳层平面波电离与辐射偏振计数。"""
from pathlib import Path
import csv
import json
import os
import sys

out = Path(__file__).resolve().parent
root = out.parent.parent
deps = root / "tmp/section3_1_deps"
if sys.version_info[:2] == (3, 12) and deps.exists():
    sys.path.insert(0, str(deps))
os.environ["MPLCONFIGDIR"] = str(root / "tmp/section5_8_mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
fontpath = Path("C:/Windows/Fonts/msyh.ttc")
if fontpath.exists():
    font_manager.fontManager.addfont(str(fontpath))
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"font.size":11,
                    "svg.fonttype":"none","axes.unicode_minus":False,
                    "axes.spines.top":False,"axes.spines.right":False})
checks = []
def check(name, residual, tolerance):
    err = float(np.max(np.abs(residual)))
    checks.append({"name":name,"residual":err,"tol":tolerance,"passed":bool(err<tolerance)})

# 矩阵元直接径向积分：单位κ=1, ℏ=1。位置导数产生j1，分部积分结果含q·F(q)。
nodes, weights = np.polynomial.legendre.leggauss(20)
r = ((np.arange(80)[:,None]+(nodes[None,:]+1)/2)*.6).ravel()
rw = np.tile(weights*.3,80)
radial = 4*np.sqrt(np.pi)*r**2*np.exp(-r)*rw
for q in [.1,1.,3.,8.,9.]:
    qr = q*r
    j0 = np.sinc(qr/np.pi)
    j1 = np.sin(qr)/qr**2-np.cos(qr)/qr
    analytic = 8*np.sqrt(np.pi)/(1+q*q)**2
    check(f"fourier_1s_q={q}", np.sum(radial*j0)-analytic,1e-11)
    check(f"velocity_integral_q={q}", np.sum(radial*j1)-q*analytic,1e-11)

# 图中只显示课文规定的平面波末态近似；横纵角明确相对于入射波的z轴及x偏振。
Z = 2
alpha = 1/137.
u = 8.  # k_f/κ
v = Z*alpha*(1+u*u)/2  # k_gamma/κ，由吸收能壳决定
A = 1+u*u+v*v
B = 2*u*v
theta = np.linspace(0, np.pi, 1001)
phi = np.linspace(0, 2*np.pi, 361)
theta_max = float(np.arccos(4*B/(A+np.sqrt(A*A+8*B*B))))
def shape(th, ph):
    return np.sin(th)**2*np.cos(ph)**2*(A/(A-B*np.cos(th)))**4
peak = float(shape(theta_max,0))
check("peak_position_vs_fine_grid",theta_max-theta[np.argmax(shape(theta,0))],.002)
check("azimuthal_node",shape(theta,np.pi/2),1e-14)
ct, wt = np.polynomial.legendre.leggauss(100)
angular_int = np.pi*np.sum(wt*(1-ct*ct)*(A/(A-B*ct))**4)
check("integrated_angular_shape", angular_int-4*np.pi/(3*(1-(B/A)**2)**2),1e-12)
with (out/"5.8-K壳层角分布切片.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f);w.writerow(["theta_deg","full_at_phi0_normalized_at_90deg",
                                "dipole_normalized_at_90deg","full_at_phi90"])
    w.writerows(zip(np.degrees(theta),shape(theta,0),np.sin(theta)**2,shape(theta,np.pi/2)))

# s←p的真正两个横偏振及角平均，径向矩阵元暂以1为单位。
mu, mw = np.polynomial.legendre.leggauss(40)
az = 2*np.pi*np.arange(80)/80
st = np.sqrt(1-mu*mu)[:,None]
ca,sa = np.cos(az)[None,:],np.sin(az)[None,:]
er=np.stack((st*ca,st*sa,mu[:,None]*np.ones_like(ca)),axis=-1)
et=np.stack((mu[:,None]*ca,mu[:,None]*sa,-st*np.ones_like(ca)),axis=-1)
ep=np.stack((-np.ones_like(st)*sa,np.ones_like(st)*ca,np.zeros_like(st*ca)),axis=-1)
w_solid=mw[:,None]*2*np.pi/80
vec={-1:np.array([1,-1j,0])/np.sqrt(6),0:np.array([0,0,1])/np.sqrt(3),
     1:-np.array([1,1j,0])/np.sqrt(6)}
amps = []
for m,d in vec.items():
    polsum=abs(et@d)**2+abs(ep@d)**2
    check(f"transverse_projector_m={m}",polsum-(np.vdot(d,d).real-abs(er@d)**2),1e-13)
    check(f"rate_m={m}",np.sum(w_solid*polsum)/(2*np.pi)-4/9,1e-13)
    amps.append(polsum)
check("unpolarized_isotropy",np.mean(amps,axis=0)-2/9,1e-13)

fig, axes = plt.subplots(1,2,figsize=(11.2,4.3),constrained_layout=True)
axes[0].plot(np.degrees(theta),shape(theta,0),lw=2.3,color="#166682",label="保留传播相位")
axes[0].plot(np.degrees(theta),np.sin(theta)**2,lw=1.6,ls="--",color="#A05232",label="电偶极近似")
axes[0].plot([np.degrees(theta_max)],[peak],"o",color="#166682",ms=5)
axes[0].annotate(f"峰位 {np.degrees(theta_max):.1f}°",
                 (np.degrees(theta_max),peak),xytext=(17,1.16),
                 arrowprops={"arrowstyle":"->","color":"#166682"},fontsize=9)
axes[0].set(xlabel=r"电子与光传播轴的夹角 $\theta$（度）",ylabel="相对微分截面",
            title=r"偏振平面内，$\phi=0$",xlim=(0,180),ylim=(0,1.25))
axes[0].set_xticks([0,45,90,135,180]);axes[0].grid(alpha=.17);axes[0].legend(fontsize=10)
surface=shape(theta[None,:],phi[:,None])
im=axes[1].pcolormesh(np.degrees(theta),np.degrees(phi),surface,shading="auto",cmap="viridis",rasterized=True)
axes[1].set(xlabel=r"电子极角 $\theta$（度）",ylabel=r"从偏振轴量的方位角 $\phi$（度）",
            title="两角分布：颜色表示每单位立体角的截面")
axes[1].set_xticks([0,45,90,135,180]);axes[1].set_yticks([0,90,180,270,360])
fig.colorbar(im,ax=axes[1],label="相对微分截面")
fig.savefig(out/"5.8-K壳层光电子两角分布.png",dpi=190)
fig.savefig(out/"5.8-K壳层光电子两角分布.svg")
plt.close(fig)
result={"purpose":"§5.8教学补充，不计章末原题；图为平面波近似的相对角分布。",
        "parameters":{"Z":Z,"u":u,"v":v,"alpha":alpha,"peak_theta_deg":float(np.degrees(theta_max)),
                      "beta_electron":Z*alpha*u,"sommerfeld_abs":1/u},
        "checks":checks,"count":len(checks),"all_passed":all(x["passed"] for x in checks)}
(out/"5.8-K壳层与偏振计数核验.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"count":len(checks),"all_passed":result["all_passed"],"parameters":result["parameters"]},
                 ensure_ascii=False))
if not result["all_passed"]:
    raise RuntimeError("有数值核验未通过，请查看JSON。")

