"""原题5.49、5.50。径向积分独立自解析氢波函数计算，角分布和偏振从矢量投影计算。
输出本目录中的 JSON、CSV 和 PNG。常数为 NIST CODATA 2022。"""
import csv
import json
import math
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent.parent
deps = PROJECT/'tmp'/'section3_1_deps'
if sys.version_info[:2] == (3, 12) and deps.is_dir():
    sys.path.insert(0, str(deps))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT/'tmp'/'section5_8_hydrogen'/'mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

PI=math.pi
ALPHA=7.2973525643e-3
ME=9.1093837139e-31
A0=5.29177210544e-11
C=299792458.
HBAR=6.62607015e-34/(2*PI)
PROTON_TO_ELECTRON=1836.152673426
REDUCTION=PROTON_TO_ELECTRON/(1+PROTON_TO_ELECTRON)

def radial(n,l,x):
    """Dimensionless radial R, normalized with x^2 dx; x=r/a."""
    if (n,l)==(1,0): return 2*np.exp(-x)
    if (n,l)==(2,0): return (2-x)*np.exp(-x/2)/(2*math.sqrt(2))
    if (n,l)==(2,1): return x*np.exp(-x/2)/(2*math.sqrt(6))
    if (n,l)==(3,1): return 4*x*(6-x)*np.exp(-x/3)/(81*math.sqrt(6))
    raise ValueError((n,l))


def transition(n_i,n_f,radial_integral):
    omega=ME*C*C*ALPHA**2*(1/n_f**2-1/n_i**2)/(2*HBAR)
    rate=(4/9)*ALPHA*omega**3/C**2*(A0*radial_integral)**2
    return {'n_i':n_i,'n_f':n_f,'R_a0':radial_integral,'omega_s-1':omega,'A_s-1':rate,
            'partial_inverse_rate_ns':1e9/rate,'A_reduced_mass_s-1':rate*REDUCTION,
            'partial_inverse_rate_reduced_mass_ns':1e9/(rate*REDUCTION)}


def make_figure():
    for candidate in ('C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf'):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams['font.family']=[font_manager.FontProperties(fname=candidate).get_name(),'DejaVu Sans']
            break
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(9.8,5.3),layout='constrained')
    grid=fig.add_gridspec(2,5,height_ratios=(2.1,1))
    ang=fig.add_subplot(grid[0,:])
    degrees=np.linspace(0,180,601)
    u=np.cos(np.radians(degrees))
    ang.plot(degrees,.75*(1+u*u),color='#245f83',lw=2.5,label='两种偏振合计')
    ang.plot(degrees,3*(1+u)**2/8,color='#b65735',lw=1.8,ls='--',label='螺旋度 +1')
    ang.plot(degrees,3*(1-u)**2/8,color='#357d60',lw=1.8,ls='-.',label='螺旋度 −1')
    ang.set(xlim=(0,180),ylim=(0,1.65),xlabel='光子方向与 +z 的夹角 θ（度）',
            ylabel=r'$(4\pi/\Gamma)\,d\Gamma/d\Omega$',title='2p，m=+1 → 1s：单位立体角辐射率')
    ang.set_xticks([0,30,60,90,120,150,180])
    ang.grid(alpha=.16)
    ang.legend(loc='upper center',ncol=3,frameon=False,fontsize=9)
    times=np.linspace(0,2*PI,300)
    for j,theta in enumerate((0,45,90,135,180)):
        ax=fig.add_subplot(grid[1,j])
        ct=np.cos(np.radians(theta))
        if np.abs(ct)<1e-12: ct=0.
        ax.axhline(0,color='#bbc0c2',lw=.6)
        ax.axvline(0,color='#bbc0c2',lw=.6)
        ax.plot(ct*np.cos(times), np.sin(times),color='#345f82',lw=1.7)
        t0,t1=.27,.80
        ax.annotate('',xy=(ct*math.cos(t1),math.sin(t1)),xytext=(ct*math.cos(t0),math.sin(t0)),
                    arrowprops={'arrowstyle':'->','color':'#ad532c','lw':2})
        ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),aspect='equal',
               title=f'{theta}°',xlabel=r'$\hat e_\theta$',xticks=[],yticks=[])
        if j==0: ax.set_ylabel(r'$\hat e_\phi$')
        ax.spines[['bottom','left']].set_visible(False)
    fig.savefig(OUT/'5.8-圆偶极角分布与偏振.png',dpi=190,bbox_inches='tight')
    plt.close(fig)


def main():
    tests=[]
    def check(label,error,tol=1e-10):
        residual=float(np.max(np.abs(error)))
        tests.append({'test':label,'residual':residual,'tolerance':tol,'passed':residual<tol})
    roots,w=np.polynomial.legendre.leggauss(160)
    x=(roots+1)*55;wx=w*55
    for n,l in [(1,0),(2,0),(2,1),(3,1)]:
        check(f'R{n}{l}归一',np.dot(wx,x*x*radial(n,l,x)**2)-1)
    transitions=[]
    radial_exact=[(2,1,256/(81*math.sqrt(6))),(3,1,81/(64*math.sqrt(6))),
                  (3,2,82944/(15625*math.sqrt(3)))]
    for ni,nf,R in radial_exact:
        numerical=np.dot(wx,x**3*radial(nf,0,x)*radial(ni,1,x))
        check(f'{ni}p→{nf}s径向积分',numerical-R)
        row=transition(ni,nf,R);row['quadrature_R_a0']=numerical;transitions.append(row)
    for m in [-1,0,1]:
        phi=2*PI*np.arange(32)/32
        z,az=np.meshgrid(roots,phi,indexing='ij');s=np.sqrt(1-z*z)
        n=np.stack([s*np.cos(az),s*np.sin(az),z],axis=-1)
        y=math.sqrt(3/(4*PI))*z if m==0 else -m*math.sqrt(3/(8*PI))*s*np.exp(1j*m*az)
        integral=(2*PI/len(phi))*np.sum(w[:,None,None]*y[:,:,None]*n,axis=(0,1))/math.sqrt(4*PI)
        theory=np.array([0,0,1/math.sqrt(3)]) if m==0 else np.array([-m,-1j,0])/math.sqrt(6)
        check(f'Y1m角积分 m={m}',integral-theory)
        check(f'三方向平方和 m={m}',np.vdot(integral,integral)-1/3)
    dipole=np.array([1,1j,0])/math.sqrt(2)
    angle_data=[]
    for theta in (0,30,45,60,90,120,135,150,180):
        t=np.radians(theta); c=np.cos(t); s=np.sin(t)
        phi=.73
        n=np.array([s*np.cos(phi),s*np.sin(phi),c])
        e_theta=np.array([c*np.cos(phi),c*np.sin(phi),-s])
        e_phi=np.array([-np.sin(phi),np.cos(phi),0])
        helicities=[(e_theta+1j*h*e_phi)/math.sqrt(2) for h in (1,-1)]
        components=np.array([np.vdot(e,dipole) for e in helicities])
        strength=np.sum(np.abs(components)**2)
        check(f'横向投影 θ={theta}',strength-(1+c*c)/2)
        check(f'局域螺旋基权重 θ={theta}',np.abs(components)**2-np.array([(1+c)**2,(1-c)**2])/4)
        check(f'Jones比 θ={theta}',
              [np.vdot(e_theta,dipole)-c*np.exp(1j*phi)/math.sqrt(2),
               np.vdot(e_phi,dipole)-1j*np.exp(1j*phi)/math.sqrt(2)])
        check(f'光子横向性 θ={theta}',np.vdot(n,dipole-n*np.vdot(n,dipole)))
        angle_data.append({'theta_deg':theta,'dGamma_dOmega_over_total':3*(1+c*c)/(16*PI),
                           'prob_hplus':float(np.abs(components[0])**2/strength),
                           'prob_hminus':float(np.abs(components[1])**2/strength),
                           'V_over_I_defined_here':float(2*c/(1+c*c))})
    check('全立体角积分等于总率',2*PI*np.dot(w,3*(1+roots**2)/(16*PI))-1)
    upper3_rate=sum(t['A_s-1'] for t in transitions[1:])
    summary={'source':'原题5.49/5.50 PDF385／印刷363；正文(5.352)—(5.359)',
             'constants_source':'https://physics.nist.gov/cuu/pdf/all.pdf','units':'角频率s^-1，A s^-1，寿命ns',
             'conventions':{'a0_m':A0,'alpha':ALPHA,'electron_mass_kg':ME,'proton_mass_over_me':PROTON_TO_ELECTRON,
                            'reduced_mass_over_me':REDUCTION,'complex_polarization':'epsilon_h=(e_theta+i h e_phi)/sqrt(2)'},
             'transitions':transitions,'total_A_3p_s-1':upper3_rate,'tau_3p_ns':1e9/upper3_rate,
             'tau_3p_reduced_ns':1e9/(REDUCTION*upper3_rate),
             'branch_3p_to_1s':transitions[1]['A_s-1']/upper3_rate,
             'historical_experiment':{'doi':'10.1103/PhysRev.148.1','table':'I','2p_mean_ns':[1.600,.004],
                                      '3p_mean_ns':[5.58,.13],'paper_theory_ns':[1.60,5.40]},
             'angular_samples':angle_data,'count':len(tests),'tests':tests,'passed':all(t['passed'] for t in tests)}
    (OUT/'5.8-氢辐射寿命与偏振核验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'5.8-氢E1辐射率与寿命.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=transitions[0].keys())
        writer.writeheader();writer.writerows(transitions)
    print(json.dumps({'passed':summary['passed'],'count':len(tests),'failed':[t for t in tests if not t['passed']],
                      'tau_3p_ns':summary['tau_3p_ns'],'branch_3p_to_1s':summary['branch_3p_to_1s']},ensure_ascii=False))
    if not summary['passed']:
        raise RuntimeError('氢辐射核验失败')
    make_figure()


if __name__=='__main__':
    main()
