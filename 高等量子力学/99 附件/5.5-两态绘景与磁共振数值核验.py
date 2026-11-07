"""§5.5 教学计算，比较原薛定谔方程、精确旋转驱动和线偏振近共振近似。

时间无量纲化为 s=γt/ℏ，采用 γ=ℏ=1。需要 numpy、matplotlib。
重新运行会在本脚本所在的附件目录更新同名图和数据。
"""
from pathlib import Path
import csv
import json
import os
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
deps = ROOT / "tmp" / "section3_1_deps"
if sys.version_info[:2] == (3, 12) and deps.exists():
    sys.path.insert(0, str(deps))
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp" / "section5_5_mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

cn_font = Path("C:/Windows/Fonts/msyh.ttc")
if cn_font.exists():
    font_manager.fontManager.addfont(str(cn_font))
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"],
                     "font.size": 11, "axes.unicode_minus": False,
                     "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False})
checks = []

def check(name, residual, tol):
    worst = float(np.max(np.abs(residual)))
    checks.append({"name": name, "residual": worst, "tolerance": tol,
                   "passed": bool(worst < tol)})

def exact_rotating_c(s, delta, initial=(1, 0)):
    """返回相互作用绘景的 c1,c2。delta=(ω−ω21)/g，g=γ/ℏ。"""
    s = np.asarray(s, dtype=float)
    q = np.sqrt(1 + (delta/2)**2)
    cos_part = np.cos(q*s)
    sin_part = np.sin(q*s)/q
    b1 = cos_part*initial[0] - 1j*sin_part*(delta*initial[0]/2 + initial[1])
    b2 = cos_part*initial[1] - 1j*sin_part*(initial[0] - delta*initial[1]/2)
    return np.column_stack((np.exp(1j*delta*s/2)*b1, np.exp(-1j*delta*s/2)*b2))

def schrodinger_rk4(s, w0, omega, drive, initial=(1, 0)):
    """直接在实验室薛定谔绘景中积分，E1/γ=−w0/2，E2/γ=+w0/2。"""
    def rhs(t, a):
        v12 = np.exp(1j*omega*t) if drive == "rotating" else 2*np.cos(omega*t)
        return -1j*np.array([-w0*a[0]/2 + v12*a[1],
                             np.conj(v12)*a[0] + w0*a[1]/2])
    a = np.empty((len(s), 2), dtype=complex)
    a[0] = initial
    for j, h in enumerate(np.diff(s)):
        t, old = s[j], a[j]
        k1 = rhs(t, old)
        k2 = rhs(t+h/2, old+h*k1/2)
        k3 = rhs(t+h/2, old+h*k2/2)
        k4 = rhs(t+h, old+h*k3)
        a[j+1] = old+h*(k1+2*k2+2*k3+k4)/6
    return a

def draw_save(fig, stem):
    fig.savefig(OUT / (stem + ".png"), dpi=195)
    fig.savefig(OUT / (stem + ".svg"))
    plt.close(fig)

def export_csv(name, header, columns):
    with (OUT/name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip(*columns))

# 旋转驱动：在原始薛定谔方程数值解上同时检验概率、相位和能量交换。
s_check = np.linspace(0, 2*np.pi, 4001)
w0 = 8.0
for delta in [0., -2., 2., 4.]:
    omega = w0+delta
    a = schrodinger_rk4(s_check, w0, omega, "rotating")
    c_numeric = a*np.column_stack((np.exp(-1j*w0*s_check/2), np.exp(1j*w0*s_check/2)))
    c_exact = exact_rotating_c(s_check, delta)
    p2 = np.abs(c_numeric[:, 1])**2
    check(f"rotating_complex_amplitudes_delta_{delta}", c_numeric-c_exact, 2e-8)
    check(f"schrodinger_norm_delta_{delta}", np.sum(np.abs(a)**2, axis=1)-1, 2e-9)
    v12 = np.exp(1j*omega*s_check)
    v_exp = 2*np.real(np.conj(a[:, 0])*v12*a[:, 1])
    check(f"drive_energy_delta_{delta}", v_exp-delta*p2, 2e-8)

psi_initial = np.array([1+2j, -3j], dtype=complex)/np.sqrt(14)
a_arbitrary = schrodinger_rk4(s_check, w0, w0+1.4, "rotating", psi_initial)
c_arbitrary = a_arbitrary*np.column_stack((np.exp(-1j*w0*s_check/2),
                                           np.exp(1j*w0*s_check/2)))
check("arbitrary_complex_initial_amplitudes",
      c_arbitrary-exact_rotating_c(s_check, 1.4, psi_initial), 2e-8)
for delta in [0., 2., -2., 4.]:
    omega_bar = np.sqrt(1+delta**2/4)
    peak_time = np.pi/(2*omega_bar)
    peak = abs(exact_rotating_c([peak_time], delta)[0, 1])**2
    check(f"first_maximum_delta_{delta}", peak - 1/omega_bar**2, 1e-12)
check("on_resonance_return_period", abs(exact_rotating_c([np.pi],0)[0,1]), 1e-12)
check("half_height_two_sides", np.array([1/(1+(-2)**2/4), 1/(1+2**2/4)])-.5, 1e-12)
check("excited_to_ground_pi_pulse", abs(exact_rotating_c([np.pi/2],0,(0,1))[0,0])**2-1, 1e-12)

# 教材图5.5与图5.6的不同量：时间轨迹、随频率取最大、固定时长扫描。
s = np.linspace(0, 2*np.pi, 801)
p2_res = np.abs(exact_rotating_c(s, 0.)[:,1])**2
p2_det = np.abs(exact_rotating_c(s, 2.)[:,1])**2
delta_grid = np.linspace(-6,6,1001)
q = np.sqrt(1+delta_grid**2/4)
envelope = 1/q**2
fixed_pulse = envelope*np.sin(q*np.pi/2)**2
check("fixed_pulse_below_envelope", np.maximum(0,fixed_pulse-envelope), 1e-12)
export_csv("5.5-拉比时间曲线.csv", ["s=g*t","P1_res","P2_res","P2_delta_over_g_2"],
           [s,1-p2_res,p2_res,p2_det])
export_csv("5.5-共振峰与定时扫描.csv", ["delta_over_g","P2_max","P2_at_gT_pi_over_2"],
           [delta_grid,envelope,fixed_pulse])
fig, ax = plt.subplots(1,2,figsize=(12.3,4.8),constrained_layout=True)
ax[0].plot(s,p2_res,lw=2.5,color="#276D91",label="共振：高能级概率")
ax[0].plot(s,1-p2_res,lw=1.7,ls="--",color="#A75B2D",label="共振：低能级概率")
ax[0].plot(s,p2_det,lw=1.9,color="#54865E",label=r"失谐 $\delta=2g$：高能级概率")
ax[0].set(xlabel=r"时间 $gt$， $g=\gamma/\hbar$",ylabel="占据概率",
          title="相同驱动强度下的时间演化",ylim=(-.035,1.06))
ax[0].set_xticks([0,np.pi/2,np.pi,1.5*np.pi,2*np.pi],
                  ["0",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
ax[0].legend(fontsize=9,loc="upper center",bbox_to_anchor=(.5,-.19),ncol=2)
ax[1].plot(delta_grid,envelope,lw=2.5,color="#276D91",label=r"各失谐分别取时间最大值 $P_{2,\max}$")
ax[1].plot(delta_grid,fixed_pulse,lw=1.9,ls="--",color="#A75B2D",
           label=r"固定作用时间 $T=\pi/(2g)$")
ax[1].plot([-2,2],[.5,.5],ls=":",lw=1.5,color="#54865E")
ax[1].plot([-2,2],[.5,.5],"o",ms=5,color="#54865E")
ax[1].annotate(r"半高全宽 $4g$",xy=(1.8,.5),xytext=(2.8,.72),
               arrowprops={"arrowstyle":"->","color":"#35543D"},fontsize=10)
ax[1].set(xlabel=r"失谐 $\delta/g=(\omega-\omega_{21})/g$",ylabel="高能级概率",
          title="改变驱动频率时，两种测量有不同曲线",ylim=(-.035,1.06))
ax[1].legend(fontsize=9,loc="upper center",bbox_to_anchor=(.5,-.19))
for a in ax:a.grid(alpha=.18)
draw_save(fig,"5.5-拉比振荡与共振峰")

# 线偏振驱动：相同单个旋转分量g，比较完整2cos(ωt)驱动、保留共转项的RWA。
s_linear = np.linspace(0,np.pi,8001)
p_rwa = np.sin(s_linear)**2
linear_data = {}
weak_and_strong = [(30., "#276D91"), (4., "#A75B2D")]
fig, ax = plt.subplots(1,2,figsize=(11.8,4.2),constrained_layout=True)
for axis,(ratio,color) in zip(ax,weak_and_strong):
    full = schrodinger_rk4(s_linear, ratio, ratio, "linear")
    coarse = schrodinger_rk4(s_linear[::2], ratio, ratio, "linear")
    p_full = np.abs(full[:,1])**2
    max_diff = float(np.max(np.abs(p_full-p_rwa)))
    linear_data[str(int(ratio))] = {"omega21_over_g":ratio,"B1_over_B0":2/ratio,
        "max_probability_difference_first_resonant_cycle":max_diff,
        "P2_at_pi_pulse_time": float(p_full[len(s_linear)//2])}
    check(f"linear_norm_w0_over_g_{ratio}",np.sum(abs(full)**2,axis=1)-1,2e-8)
    check(f"linear_step_halving_w0_over_g_{ratio}", full[::2]-coarse, 2e-8)
    export_csv(f"5.5-线偏振核验-频率比{int(ratio)}.csv",
               ["g*t","P2_full_linear","P2_RWA"],[s_linear[::10],p_full[::10],p_rwa[::10]])
    axis.plot(s_linear,p_full,color=color,lw=2.25,label="完整线偏振场")
    axis.plot(s_linear,p_rwa,color="#30373A",ls="--",lw=1.6,label="只保留共转分量")
    axis.set_title(rf"$\omega_{{21}}/g={ratio:.0f}$，$B_1/B_0={2/ratio:.3f}$")
    axis.set(xlabel=r"共振驱动的持续时间 $gt$",ylabel="高能级概率",ylim=(-.035,1.06))
    axis.set_xticks([0,np.pi/2,np.pi],["0",r"$\pi/2$",r"$\pi$"])
    axis.legend(fontsize=9,loc="lower center")
    axis.grid(alpha=.18)
fig.suptitle("线偏振场含两个反向旋转分量",fontsize=15)
draw_save(fig,"5.5-线偏振场与旋转波近似")

summary = {"units":"s=γt/ℏ, delta/g, g=γ/ℏ; hbar=g=1 in numerical equations",
    "example":"教学计算，不计教材原题",
    "resonance":{"first_transfer_g_t":np.pi/2,"return_g_t":np.pi,"FWHM_over_g":4,
                 "delta_2g_peak":.5,"delta_2g_fixed_pi_pulse":float(.5*np.sin(np.pi/np.sqrt(2))**2)},
    "linear_drive_comparison":linear_data,"checks":checks,
    "check_count":len(checks),"all_passed":all(row["passed"] for row in checks)}
(OUT/"5.5-两态绘景与磁共振数值核验.json").write_text(
    json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"checks":len(checks),"all_passed":summary["all_passed"],
                  "linear":linear_data,"fixed_pulse_at_2g":summary["resonance"]["delta_2g_fixed_pi_pulse"]},
                 ensure_ascii=False,indent=2))
if not summary["all_passed"]:
    raise RuntimeError("核验失败，请核对 JSON 中的误差。")

