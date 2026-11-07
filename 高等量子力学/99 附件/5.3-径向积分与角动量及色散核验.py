"""§5.3 教学补充：独立数值积分、角动量矩阵、束缚谱色散和。
不改教材、不计为章末原题；不重复5.24八态图。
依赖numpy、matplotlib。长度a0=1、能量Eh=1；中间场图另取A=1。
"""
from pathlib import Path
import sys, os, math, json, csv
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
local = ROOT / 'tmp/section3_1_deps'
if sys.version_info[:2] == (3,12) and local.exists():
    sys.path.insert(0, str(local))
os.environ['MPLCONFIGDIR'] = str(ROOT/'tmp/section5_3_calculations/mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],
                    'font.size':11, 'axes.unicode_minus':False,
                    'svg.fonttype':'none', 'axes.spines.top':False,
                    'axes.spines.right':False})
checks=[]
def check(name, error, tol=1e-9):
    err=float(np.max(np.abs(error)))
    checks.append({'name':name,'error':err,'tolerance':tol,'passed':bool(err<tol)})

# Gauss-Laguerre直接积掉e^-rho，使用完整归一化径向波函数。
rho, weight=np.polynomial.laguerre.laggauss(64)
def radial_moment(n,l,s):
    q=n-l-1
    lag=sum((-1)**i*math.comb(n+l,q-i)*rho**i/math.factorial(i) for i in range(q+1))
    c=math.factorial(q)/(2*n*math.factorial(n+l))
    return (n/2)**s*c*np.sum(weight*rho**(2*l+s+2)*lag**2)
radial=[]
for n in range(1,7):
    for l in range(n):
        targets={0:1.,-1:1/n**2,-2:1/(n**3*(l+.5))}
        if l:
            targets[-3]=1/(n**3*l*(l+.5)*(l+1))
        vals={}
        for s,exact in targets.items():
            val=radial_moment(n,l,s)
            vals[str(s)]=float(val)
            check(f'radial_n{n}_l{l}_s{s}',val-exact)
        boundary=0 if l else 2/n**3
        lhs=vals['-2']-(l*(l+1)*vals.get('-3',0))
        check(f'origin_boundary_n{n}_l{l}',lhs-boundary)
        radial.append({'n':n,'l':l,'moments':vals,'origin_force_term':boundary})

# 5.103的有效角动量解，数值微分与直接径向积分互证。
hf=[]
for n,l in [(1,0),(2,0),(2,1),(4,2)]:
    nr=n-l-1
    def energy_gamma(g):
        ell=-.5+math.sqrt((l+.5)**2+2*g)
        return -.5/(nr+ell+1)**2
    h=1e-6
    derivative=(energy_gamma(h)-energy_gamma(-h))/(2*h)
    check(f'hellmann_feynman_rminus2_n{n}_l{l}',derivative-radial_moment(n,l,-2),2e-8)
    hf.append({'n':n,'l':l,'derivative':derivative})

# 独立构造完整(LxSx+LySy+LzSz)+(Lz+2Sz)b，再抽取固定m块。
def operators(l):
    ml=np.arange(l,-l-1,-1,dtype=float)
    lp=np.zeros((2*l+1,2*l+1),complex)
    for j,m in enumerate(ml):
        if j>0:
            lp[j-1,j]=math.sqrt(l*(l+1)-m*(m+1))
    lx=(lp+lp.conj().T)/2
    ly=(lp-lp.conj().T)/(2j)
    lz=np.diag(ml)
    sx=np.array([[0,1],[1,0]],complex)/2
    sy=np.array([[0,-1j],[1j,0]],complex)/2
    sz=np.diag([.5,-.5])
    ls=np.kron(lx,sx)+np.kron(ly,sy)+np.kron(lz,sz)
    z=np.kron(lz,np.eye(2))+2*np.kron(np.eye(2*l+1),sz)
    basis=[(a,b) for a in ml for b in [.5,-.5]]
    return ls,z,basis
for l in range(1,5):
    ls,z,basis=operators(l)
    for m in np.arange(-l-.5,l+.51,1):
        idx=[i for i,(ml,ms) in enumerate(basis) if abs(ml+ms-m)<1e-10]
        for b in [0,.05,.5,2,10]:
            numerical=np.linalg.eigvalsh((ls+b*z)[np.ix_(idx,idx)])
            if len(idx)==1:
                exact=np.array([l/2+np.sign(m)*b*(l+1)])
            else:
                d=math.sqrt((l+.5)**2+2*m*b+b*b)
                exact=np.array([-1/4+b*m-d/2,-1/4+b*m+d/2])
            check(f'lande_l{l}_m{m}_b{b}',numerical-exact)

# 教学l=2,m=-1/2块：显示有限场态混合，而非5.24的氢n=2八态图。
l,m=2,-.5
b=np.linspace(0,8,401)
d=np.sqrt((l+.5)**2+2*m*b+b*b)
emin=-.25+b*m-d/2
emax=-.25+b*m+d/2
pup_low=(1-(m+b)/d)/2
sz_low=pup_low-.5
lande_rows=[]
for b0 in [0,.5,2,8]:
    d0=math.sqrt((l+.5)**2+2*m*b0+b0*b0)
    lande_rows.append({'b_over_A':b0,'E_low_over_A':-.25+b0*m-d0/2,
                      'E_high_over_A':-.25+b0*m+d0/2,
                      'P_up_low':(1-(m+b0)/d0)/2,
                      'Sz_low_over_hbar':-(m+b0)/(2*d0)})
with (OUT/'5.3-固定m中间场混合.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['b/A','E_low/A','E_high/A','P_up_low','Sz_low/hbar'])
    w.writerows(zip(b,emin,emax,pup_low,sz_low))
fig,ax=plt.subplots(1,2,figsize=(12.5,4.7),constrained_layout=True)
ax[0].plot(b,emin,label='低能支（B=0时 j=3/2）',lw=2.3)
ax[0].plot(b,emax,label='高能支（B=0时 j=5/2）',lw=2.3)
ax[0].plot(b,-b,color='0.5',ls=':',label='强场乘积态对角能量',lw=1.6)
ax[0].plot(b,np.zeros_like(b)-.5,color='0.5',ls=':',lw=1.6)
ax[0].set(xlabel=r'$b/A=\mu_B B/A$',ylabel=r'能量 $E/A$',title=r'固定 $l=2,\ m=-1/2$：两态能量')
ax[0].legend(fontsize=9,loc='lower left')
ax[1].plot(b,pup_low,label=r'低能支自旋向上概率 $P_\uparrow$',lw=2.3)
ax[1].plot(b,sz_low,label=r'低能支 $\langle S_z\rangle/\hbar$',lw=2.3)
ax[1].axhline(0,color='.6',lw=.8)
ax[1].axvline(.5,color='.6',ls=':',lw=1,label='两对角能量相等')
ax[1].set(xlabel=r'$b/A=\mu_B B/A$',ylabel='概率／无量纲期望值',title='本征态连续旋转，不能只换能量公式')
ax[1].legend(fontsize=9)
for a in ax:a.grid(alpha=.18)
fig.suptitle('§5.3 教学补充：弱场耦合基到强场乘积基',fontsize=16)
fig.savefig(OUT/'5.3-固定m中间场混合.png',dpi=190)
fig.savefig(OUT/'5.3-固定m中间场混合.svg')
plt.close(fig)

# 1s->np离散束缚谱：完整C6还必须加一/两个原子处连续谱的贡献。
def bound_bound_c6(nmax):
    ns=np.arange(2,nmax+1,dtype=float)
    logf=np.log(256/3)+5*np.log(ns)-8*np.log(ns+1)+(2*ns-4)*np.log1p(-2/(ns+1))
    f=np.exp(logf)
    delta=.5*(1-1/ns**2)
    radial2=3*f/(1-1/ns**2)
    value=(2/3)*np.sum(radial2[:,None]*radial2[None,:]/(delta[:,None]+delta[None,:]))
    return float(value),float(np.sum(f))
nmaxs=[2,3,4,5,10,20,50,100,300,1000]
dispersion=[]
last=0
for nmax in nmaxs:
    val,fsum=bound_bound_c6(nmax)
    check(f'bound_partial_monotone_N{nmax}',0 if val>last else 1)
    check(f'bound_partial_below_rigorous_lower_N{nmax}',0 if val<6 else 1)
    dispersion.append({'nmax':nmax,'C6_bound_bound_Eh_a0_6':val,'discrete_oscillator_strength_sum':fsum})
    last=val
check('2p_radial_integral',math.sqrt(3*(2**13/3**9)/(1-.25))-256/(81*math.sqrt(6)))
with (OUT/'5.3-范德瓦耳斯束缚谱部分和.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=dispersion[0].keys());w.writeheader();w.writerows(dispersion)
fig,ax=plt.subplots(figsize=(9.8,5.2),constrained_layout=True)
ax.axhspan(6,8,color='#B1D9C9',alpha=.65,label='完整非迟滞 C₆ 的自推严格界：6 ≤ C₆ ≤ 8')
ax.axhline(6,color='#39715D',lw=1.3)
ax.axhline(8,color='#39715D',lw=1.3)
ax.plot(nmaxs,[r['C6_bound_bound_Eh_a0_6'] for r in dispersion],'o-',color='#C86B39',lw=2.3,label='只含两原子束缚 np 态的正项和')
ax.text(2.3,4.65,'离散部分和已稳定，仍漏掉连续谱\n这条曲线不能当作完整 C₆',color='#8B4624',fontsize=11)
ax.set(xscale='log',xlabel='每个原子的束缚态截断 n ≤ N',ylabel=r'$C_6/(E_h a_0^6)$',ylim=(2,8.6),title='§5.3 教学补充：级数“稳定”不等于完备性成立')
ax.legend(loc='upper center',fontsize=10)
ax.grid(alpha=.18)
fig.savefig(OUT/'5.3-范德瓦耳斯束缚谱与完整性.png',dpi=190)
fig.savefig(OUT/'5.3-范德瓦耳斯束缚谱与完整性.svg')
plt.close(fig)

result={'date':'2026-10-30','purpose':'教学补充，不计教材原题',
        'radial':radial,'hellmann_feynman':hf,'intermediate_field':lande_rows,
        'dispersion_bound_bound':dispersion,
        'rigorous_full_C6_bounds_Eh_a0_6':[6,8],
        'warning':'无完整连续谱计算，不报告精确C6；N=1000仅为束缚-束缚部分和。',
        'checks':checks,'check_count':len(checks),'all_passed':all(x['passed'] for x in checks)}
(OUT/'5.3-径向积分与角动量及色散核验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'checks':len(checks),'all_passed':result['all_passed'],'C6_bound_bound_N1000':dispersion[-1],
                  'intermediate_field':lande_rows},ensure_ascii=False,indent=2))
if not result['all_passed']:
    raise SystemExit('Some checks failed; inspect JSON.')
