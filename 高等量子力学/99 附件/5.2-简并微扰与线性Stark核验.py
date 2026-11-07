"""§5.2教学补充：n=2 Stark积分、投影有效矩阵、近简并交叉。

所有数值参数均为教学自选；不写入教材，不计作教材原题。
依赖numpy、matplotlib。可使用Codex bundled Python 3.12运行。
"""
from pathlib import Path
import sys
import os
import json
import csv

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
if sys.version_info[:2] == (3, 12):
    local = ROOT / 'tmp/section3_1_deps'
    if local.exists():
        sys.path.insert(0, str(local))
os.environ['MPLCONFIGDIR'] = str(ROOT / 'tmp/section5_2_calculations/mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
                     'font.size': 11, 'axes.unicode_minus': False,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})
checks = []
def check(name, residual, tolerance=1e-10):
    value = float(np.max(np.abs(residual)))
    checks.append(dict(name=name, residual=value, tolerance=tolerance,
                       passed=bool(value < tolerance)))

# 原子单位a0=1，基顺序2s,2p0,2p+1,2p-1。
# 用独立径向Gauss-Laguerre与角向Gauss-Legendre/周期求和求完整矩阵。
rho, rw = np.polynomial.laguerre.laggauss(48)
polys = np.array([(2-rho)/(2*np.sqrt(2))] +
                 [rho/(2*np.sqrt(6))]*3)
mu, mw = np.polynomial.legendre.leggauss(48)
phi = 2*np.pi*np.arange(64)/64
ones = np.ones((len(mu), len(phi)))
ys = np.array([
    ones/np.sqrt(4*np.pi),
    np.sqrt(3/(4*np.pi))*mu[:,None]*ones,
    -np.sqrt(3/(8*np.pi))*np.sqrt(1-mu[:,None]**2)*np.exp(1j*phi)[None,:],
    np.sqrt(3/(8*np.pi))*np.sqrt(1-mu[:,None]**2)*np.exp(-1j*phi)[None,:],
], dtype=complex)
angular_weight = mw[:,None]*2*np.pi/len(phi)
gram = np.zeros((4,4),complex)
z = np.zeros((4,4),complex)
angular_z = np.zeros((4,4),complex)
radial_z = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        rg = np.sum(rw*rho**2*polys[i]*polys[j])
        ag = np.sum(angular_weight*ys[i].conj()*ys[j])
        gram[i,j] = rg*ag
        radial_z[i,j] = np.sum(rw*rho**3*polys[i]*polys[j])
        angular_z[i,j] = np.sum(angular_weight*ys[i].conj()*mu[:,None]*ys[j])
        z[i,j] = radial_z[i,j]*angular_z[i,j]
z_expected = np.zeros((4,4)); z_expected[0,1]=z_expected[1,0]=-3
check('n2基四态正交归一', gram-np.eye(4))
check('径向2s-2p积分=-3sqrt3 a0', radial_z[0,1]+3*np.sqrt(3))
check('角向Y00-cos-Y10积分=1/sqrt3', angular_z[0,1]-1/np.sqrt(3))
check('完整z矩阵含全部零元', z-z_expected)
field = .0001
charge = -1.
v = -charge*field*z
energies, states = np.linalg.eigh(v)
check('n2线性Stark四个能移', energies-np.array([-3*field,0,0,3*field]))
sym = np.array([1,1,0,0])/np.sqrt(2)
anti = np.array([1,-1,0,0])/np.sqrt(2)
check('对称态是e<0时的低能态', v@sym+3*field*sym)
check('反对称态是e<0时的高能态', v@anti-3*field*anti)
check('低能态沿场偶极=+3', charge*(sym@z@sym)-3)
check('高能态逆场偶极=-3', charge*(anti@z@anti)+3)
u = np.diag([1,np.exp(.731j),1,1])
zp = u.conj().T@z@u
symp = u.conj().T@sym
check('基相位变化不改变偶极', symp.conj()@zp@symp - sym@z@sym)

# 一个自选三态示例，验证P内一阶态修正不能遗漏。
# H0=diag(0,0,3)，PVP=diag(-1,+1)，P-Q耦合a=.4,b=.7。
gap=3.; a=.4; b=.7
h0=np.diag([0.,0.,gap])
vv=np.array([[-1.,0.,a],[0.,1.,b],[a,b,.2]])
bmat=-np.array([[a*a,a*b],[a*b,b*b]])/gap
chi=np.array([0., a*b/(2*gap),-a/gap])
toy=[]
for lam in [.04,.02,.01]:
    h=h0+lam*vv
    ev,vec=np.linalg.eigh(h)
    idx=np.argmin(np.abs(ev+lam))
    exact=vec[:,idx]/vec[0,idx]
    approx=np.array([1.,0.,0.])+lam*chi
    # 通过精确Q内消元验证有效矩阵（而非重复同一展开式）。
    qres=1/(ev[idx]-gap-lam*vv[2,2])
    heff=lam*vv[:2,:2]+lam**2*vv[:2,2,None]*qres*vv[None,2,:2]
    check(f'精确Q消元残差lambda={lam}',heff@exact[:2]-ev[idx]*exact[:2])
    toy.append(dict(lam=lam, energy=float(ev[idx]),
                    energy_order2=float(-lam+lam*lam*bmat[0,0]),
                    internal_exact=float(exact[1]),
                    internal_order1=float(lam*chi[1]),
                    external_exact=float(exact[2]),
                    external_order1=float(lam*chi[2]),
                    state_error=float(np.linalg.norm(exact-approx))))
ratio=toy[1]['state_error']/toy[2]['state_error']
check('一阶态误差减半参数后约降为四分之一',ratio-4.,.12)

# 残留简并：令PVP=0，二阶B的两个本征值为-(a²+b²)/gap和0。
residual_b = -np.outer([a,b],[a,b])/gap
check('残留简并二阶有效矩阵本征值',
      np.linalg.eigvalsh(residual_b)-np.array([-(a*a+b*b)/gap,0.]))

# 近简并2s/p示意：delta>0，g=3|e|a0 E，H-barE=[[delta/2,-g],[-g,-delta/2]]。
# delta取1仅为无量纲教学坐标，不冒称真实氢的兰姆位移数值。
x=np.linspace(0,4,401)
r=np.sqrt(.25+x*x)
ps=.5*(1-1/(2*r))
dip=x/r
samples=[]
for xx in [.01,.1,.5,1.,3.]:
    hh=np.array([[.5,-xx],[-xx,-.5]])
    ev,ve=np.linalg.eigh(hh)
    rr=np.sqrt(.25+xx*xx)
    exact_ps=abs(ve[0,0])**2
    check(f'近简并精确能量x={xx}',ev-np.array([-rr,rr]))
    check(f'近简并原2s概率x={xx}',exact_ps-.5*(1-1/(2*rr)))
    sigx=np.array([[0.,1.],[1.,0.]])
    exact_d=ve[:,0]@sigx@ve[:,0]
    check(f'近简并偶极期望x={xx}',exact_d-xx/rr)
    eps=1e-6
    numder=-(np.linalg.eigvalsh(np.array([[.5,-xx-eps],[-xx-eps,-.5]]))[0]
             -np.linalg.eigvalsh(np.array([[.5,-xx+eps],[-xx+eps,-.5]]))[0])/(2*eps)
    check(f'近简并能量导数核对偶极x={xx}',numder-exact_d,1e-8)
    samples.append(dict(g_over_delta=xx, shift_lower_over_delta=float(.5-rr),
                        prob_s_lower=float(exact_ps), dipole_over_kappa=float(exact_d)))

fig,ax=plt.subplots(1,2,figsize=(13.0,4.8),layout='constrained')
blue='#176b7d'; orange='#c45127'; grey='#777777'
ax[0].plot(x,-r,color=blue,lw=2.4,label='精确低能支')
ax[0].plot(x,r,color=orange,lw=2.4,label='精确高能支')
mask=x<=.6
ax[0].plot(x[mask],-(.5+x[mask]**2),'--',color=blue,lw=1.6,label='非简并二阶（仅画小场段）')
ax[0].plot(x[mask],.5+x[mask]**2,'--',color=orange,lw=1.6)
ax[0].plot(x,-x,':',color=grey,lw=1.4,label='忽略原能隙：线性 ±g')
ax[0].plot(x,x,':',color=grey,lw=1.4)
ax[0].set(xlabel='场耦合 / 原能隙  g / δ',ylabel='相对平均值的能量 / δ',
          title='小场二次响应连续转为近线性劈裂')
ax[0].legend(fontsize=9,loc='upper left')
ax[1].plot(x,ps,color=blue,lw=2.4,label='低能态中 2s 的概率')
ax[1].plot(x,1-ps,color=orange,lw=2.0,label='低能态中 2p₀ 的概率')
ax[1].plot(x,dip,'--',color=grey,lw=2.0,label='沿场偶极 / (3|e|a₀)')
ax[1].axhline(.5,color='#aaaaaa',ls=':',lw=1)
ax[1].set(xlabel='场耦合 / 原能隙  g / δ',ylabel='概率或无量纲偶极',ylim=(-.02,1.03),
          title='从一个旧态逐渐混成近乎等权叠加')
ax[1].legend(fontsize=9,loc='center right')
for aa in ax: aa.grid(alpha=.15)
fig.suptitle('近简并教学模型：δ > 0，g = 3|e|a₀ℰ；完整氢原子的弱外耦合条件另须满足',fontsize=12)
stem='5.2-近简并二次与线性Stark过渡'
fig.savefig(OUT/(stem+'.png'),dpi=190,bbox_inches='tight')
fig.savefig(OUT/(stem+'.svg'),bbox_inches='tight')
plt.close(fig)
with (OUT/(stem+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['g_over_delta','E_lower_minus_mean_over_delta','E_upper_minus_mean_over_delta',
                     'prob_s_lower','prob_p_lower','dipole_over_3abs_e_a0'])
    writer.writerows(zip(x,-r,r,ps,1-ps,dip))
report=dict(date='2026-10-30',purpose='教学补充，不计教材原题',
            hydrogen_n2=dict(basis=['2s','2p_m0','2p_m+1','2p_m-1'],
                              a0=1,radial_sp=float(radial_z[0,1]),
                              angular_sp=float(angular_z[0,1].real),
                              z_matrix_real=z.real.tolist(),
                              z_matrix_max_imag=float(np.max(np.abs(z.imag))),
                              field=field,electron_charge=charge,
                              linear_shifts=energies.tolist()),
            internal_correction_toy=toy,near_degenerate_samples=samples,
            checks=checks,all_passed=all(c['passed'] for c in checks))
(OUT/'5.2-简并微扰与线性Stark核验结果.json').write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(checks=len(checks),all_passed=report['all_passed'],
                     toy_error_ratio=ratio,near_degenerate_samples=samples),ensure_ascii=False,indent=2))
if not report['all_passed']:
    raise SystemExit('Verification failed')
