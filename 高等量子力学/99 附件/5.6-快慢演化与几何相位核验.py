"""§5.6教学核验：恒定能隙自旋的有限速转动、几何相位与磁通。
对照实验室坐标的薛定谔方程、旋转坐标的精确解、离散路径重叠。
以下参数仅供教学；不使用或拟合图5.7实验数据。依赖numpy、matplotlib。
"""
from pathlib import Path
import csv
import json
import os
import sys
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
deps = ROOT / 'tmp/section3_1_deps'
if sys.version_info[:2] == (3,12) and deps.exists():
    sys.path.insert(0,str(deps))
os.environ['MPLCONFIGDIR'] = str(ROOT/'tmp/section5_6_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
cn_font = Path('C:/Windows/Fonts/msyh.ttc')
if cn_font.is_file():
    font_manager.fontManager.addfont(str(cn_font))
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],
    'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})
checks = []
def check(name, residual, tolerance):
    error = float(np.max(np.abs(residual)))
    checks.append(dict(name=name,residual=error,tolerance=tolerance,passed=bool(error<tolerance)))

def wrap(angle):
    return np.angle(np.exp(1j*angle))

def csv_file(name, header, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def save_plot(fig, stem):
    fig.savefig(OUT/(stem+'.png'),dpi=185)
    fig.savefig(OUT/(stem+'.svg'))
    plt.close(fig)

def sigma_dot(v):
    x,y,z=v
    return np.array([[z,x-1j*y],[x+1j*y,-z]],complex)

def exp_plus_pauli(v):
    """exp(+i v·σ/2)，用无量纲三分量向量。"""
    mag=np.linalg.norm(v)
    return np.cos(mag/2)*np.eye(2) + .5j*np.sinc(mag/(2*np.pi))*sigma_dot(v)

def lab_state(k, direction, initial, steps=4096):
    """s=t/T，直接解 i dψ/ds=−(k/2)n(s)·σ ψ。"""
    step=1/steps
    state=np.asarray(initial,dtype=complex).copy()
    def rhs(s, psi):
        return .5j*k*(sigma_dot(direction(s))@psi)
    for i in range(steps):
        s=i*step
        f1=rhs(s,state)
        f2=rhs(s+step/2,state+step*f1/2)
        f3=rhs(s+step/2,state+step*f2/2)
        f4=rhs(s+step,state+step*f3)
        state+=step*(f1+2*f2+2*f3+f4)/6
    return state

alpha=np.pi/2
up=np.array([1,0],complex)
down=np.array([0,1],complex)
def meridian_leak(k, s=1.):
    q=np.hypot(k,alpha)
    return (alpha/q*np.sin(q*s/2))**2
def meridian_direction(s):
    return np.array([np.sin(alpha*s),0.,np.cos(alpha*s)])
yrot=exp_plus_pauli(np.array([0.,-alpha,0.]))
meridian_rows=[]
for k in [.05,.5,3.,10.,30.]:
    exact=yrot @ exp_plus_pauli(np.array([0.,alpha,k])) @ up
    lab=lab_state(k,meridian_direction,up)
    p=abs(np.vdot(yrot@down,lab))**2
    check(f'meridian_all_complex_components_k{k}',lab-exact,2e-8)
    check(f'meridian_instantaneous_transition_k{k}',p-meridian_leak(k),2e-8)
    check(f'meridian_norm_k{k}',np.vdot(lab,lab)-1,2e-8)
    meridian_rows.append(dict(k=k,exact_transition=float(meridian_leak(k)),
                             lab_transition=float(p),frozen_basis_survival=float(abs(lab[0])**2)))
check('sudden_projection_limit',meridian_leak(1e-7)-np.sin(alpha/2)**2,1e-13)
k_grid=np.geomspace(.035,60.,850)
p_grid=meridian_leak(k_grid)
upper=alpha**2/(k_grid**2+alpha**2)
check('transition_below_adiabatic_envelope',np.maximum(0,p_grid-upper),1e-12)
csv_file('5.6-快慢转动末态数据.csv',['omega0*T','P_final_opposite_branch','upper_envelope'],
         zip(k_grid,p_grid,upper))
fig,ax=plt.subplots(1,2,figsize=(11.9,4.25),constrained_layout=True)
ax[0].semilogx(k_grid,p_grid,lw=2.1,color='#236583',label='精确末态概率')
ax[0].semilogx(k_grid,upper,lw=1.5,ls='--',color='#958264',label='精确振荡包络')
ax[0].axhline(.5,color='#9B4A36',ls=':',lw=1.5,label='瞬变投影值 1/2')
ax[0].set(xlabel=r'总转动时间 $\omega_0 T$',ylabel='末态测到相反瞬时分支的概率',
          title='同一路径，改变走完路径的时间',ylim=(-.025,1.035))
ax[0].legend(fontsize=9)
s_grid=np.linspace(0,1,481)
for k,c in [(.1,'#9B4A36'),(3.,'#39734C'),(20.,'#236583')]:
    ax[1].plot(s_grid,meridian_leak(k,s_grid),lw=2.1,color=c,label=rf'$\omega_0T={k:g}$')
ax[1].plot(s_grid,np.sin(alpha*s_grid/2)**2,'k:',lw=1.2,label='瞬变近似的投影')
ax[1].set(xlabel=r'行程 $s=t/T$',ylabel='随行程测到相反瞬时分支的概率',
          title='磁场方向从 z 轴转向 x 轴',ylim=(-.012,.52))
ax[1].legend(fontsize=9,loc='upper left')
for a in ax:a.grid(alpha=.16)
save_plot(fig,'5.6-瞬变投影与绝热跟随')

theta=np.pi/3
plus=np.array([np.cos(theta/2),np.sin(theta/2)],complex)
minus=np.array([-np.sin(theta/2),np.cos(theta/2)],complex)
def cone_result(k, sense=1, branch=1):
    """一周返回振幅、漏出初始分支概率、扣除瞬时分支动力学相位后的相位。"""
    v=sense*2*np.pi
    q=np.sqrt(k*k+v*v+2*k*v*np.cos(theta))
    amp=-(np.cos(q/2)+1j*branch*(k+v*np.cos(theta))/q*np.sin(q/2))
    p=(v*np.sin(theta)/q*np.sin(q/2))**2
    rel=wrap(np.angle(amp)-branch*k/2)
    return amp,p,rel
cone_rows=[]
for k in [10.,30.,100.,300.]:
    amp,p,rem=cone_result(k)
    cone_rows.append(dict(k=k,leakage=float(p),relative_phase_rad=float(rem),
                    adiabatic_berry_rad=float(-np.pi*(1-np.cos(theta)))))
for k in [3.,10.,30.]:
    for sense in [-1,1]:
        v=sense*2*np.pi
        def cone_direction(s):
            return [np.sin(theta)*np.cos(v*s),np.sin(theta)*np.sin(v*s),np.cos(theta)]
        for branch,spinor in [(1,plus),(-1,minus)]:
            exact=-exp_plus_pauli(np.array([k*np.sin(theta),0.,k*np.cos(theta)+v]))@spinor
            lab=lab_state(k,cone_direction,spinor,steps=2048)
            amp,p,rem=cone_result(k,sense,branch)
            check(f'cone_state_{k}_{sense}_{branch}',lab-exact,2e-8)
            check(f'cone_overlap_{k}_{sense}_{branch}',np.vdot(spinor,lab)-amp,2e-8)
            check(f'cone_transition_{k}_{sense}_{branch}',1-abs(amp)**2-p,1e-12)

# 离散Wilson环：闭合重叠的乘积对每一格相位不敏感。
def closed_link_phase(spinors):
    overlaps=np.sum(np.conj(spinors)*np.roll(spinors,-1,axis=0),axis=1)
    if np.any(abs(overlaps)<1e-10):
        raise ValueError('网格过粗，相邻态近乎正交。')
    return -np.angle(np.prod(overlaps/abs(overlaps)))
rng=np.random.default_rng(56)
wilson_rows=[]
for colat in [np.pi/6,np.pi/3,2*np.pi/3]:
    for sense in [-1,1]:
        phi=sense*2*np.pi*np.arange(8192)/8192
        ones=np.ones_like(phi)
        for branch in [-1,1]:
            if branch==1:
                spinors=np.column_stack((np.cos(colat/2)*ones,np.exp(1j*phi)*np.sin(colat/2)))
            else:
                spinors=np.column_stack((-np.exp(-1j*phi)*np.sin(colat/2),np.cos(colat/2)*ones))
            predicted=-branch*sense*np.pi*(1-np.cos(colat))
            actual=closed_link_phase(spinors)
            rephased=closed_link_phase(spinors*np.exp(1j*rng.uniform(-np.pi,np.pi,len(phi)))[:,None])
            check(f'Wilson_loop_{colat}_{sense}_{branch}',wrap(actual-predicted),3e-7)
            check(f'Wilson_local_rephasing_{colat}_{sense}_{branch}',wrap(rephased-actual),3e-12)
            wilson_rows.append(dict(colat_rad=float(colat),sense=sense,branch=branch,
                          computed_rad=float(actual),predicted_mod_2pi=float(wrap(predicted))))

kg=np.geomspace(9,420,1801)
physical_rows=[]
fig,ax=plt.subplots(1,2,figsize=(11.7,4.25),constrained_layout=True)
for sense,c,title in [(1,'#236583','正向绕行'),(-1,'#9B4A36','反向绕行')]:
    amp,p,rem=cone_result(kg,sense)
    expected=-sense*np.pi*(1-np.cos(theta))
    ax[0].semilogx(kg,rem,lw=2,color=c,label=title+'：精确末态')
    ax[0].axhline(expected,color=c,ls='--',lw=1.2,label=title+'：贝里相位')
    v=sense*2*np.pi
    env=v**2*np.sin(theta)**2/(kg**2+v*v+2*kg*v*np.cos(theta))
    if sense==1:
        mask=kg<=100
        ax[1].loglog(kg[mask],np.maximum(p[mask],1e-12),lw=1.6,color=c,label='精确末态跃迁概率')
        ax[1].loglog(kg[mask],env[mask],color='#958264',ls='--',lw=1.5,label='跃迁概率上包络')
    physical_rows.extend(zip(kg,np.full(len(kg),sense),p,rem,np.full(len(kg),expected)))
ax[0].set(xlabel=r'一周时间 $\omega_0T$',ylabel='相位（弧度）',
          title='初态分支返回振幅扣除参考动力学相位')
ax[1].set(xlabel=r'一周时间 $\omega_0T$',ylabel='末态测到相反分支的概率',
          title='正向绕行：跃迁概率及其包络',ylim=(1e-6,1))
ax[0].legend(fontsize=8,loc='center right')
ax[1].legend(fontsize=9)
for a in ax:a.grid(alpha=.16)
save_plot(fig,'5.6-锥面绕行的相位与跃迁')
csv_file('5.6-自旋闭路有限速率数据.csv',['omega0*T','sense','leakage','relative_phase_rad','Berry_rad'],
         physical_rows)

# AB磁通线外，A=Phi(-y,x)/(2pi*r²)，用折线中点数值积分检验路径形变及反向。
ab_rows=[]
for winding in [1,-1,2]:
    for ecc in [0.,.28]:
        phi=np.linspace(0,2*np.pi*winding,8193)
        rad=2*(1+ecc*np.cos(3*phi))
        pos=np.column_stack((rad*np.cos(phi),rad*np.sin(phi)))
        midpoint=(pos[:-1]+pos[1:])/2
        dl=np.diff(pos,axis=0)
        reduced_A=np.column_stack((-midpoint[:,1],midpoint[:,0]))/(2*np.pi*np.sum(midpoint**2,axis=1)[:,None])
        reduced_line=float(np.sum(reduced_A*dl))
        check(f'AB_winding_{winding}_{ecc}',reduced_line-winding,2e-6)
        ab_rows.append(dict(winding=winding,radial_deformation=ecc,
                            line_integral_over_flux=reduced_line,
                            phase_over_pi_for_negative_charge_flux_0_4_Φ0=-.8*reduced_line))
summary={'units':'k=ω0 T, s=t/T, instantaneous gap ℏω0, theta=π/3 for cone',
         'educational_data':True,'meridian':meridian_rows,'cone':cone_rows,'wilson':wilson_rows,
         'AB':ab_rows,'checks':checks,'check_count':len(checks),
         'all_passed':all(c['passed'] for c in checks)}
(OUT/'5.6-快慢演化与几何相位核验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'checks':summary['check_count'],'all_passed':summary['all_passed'],'cone':cone_rows},ensure_ascii=False))
if not summary['all_passed']:
    raise RuntimeError('核验失败；查看JSON中的误差。')

