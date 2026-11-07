"""§4.2 教学补充：使用原题4.6核验的真实矩形双阱数据作图。

不重做求根；读同目录4.2-双势阱劈裂核验.json及本征态CSV。
参数hbar=m=b=1,V0=100,a=.4为教学选择，非原题指定数值。
另核验二能级相位、宇称分布、半空间概率及2s/2p偶极矩。
"""
from pathlib import Path
import json, os, sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(WORK/'tmp/pdfs/section4_2/mpl'))
try:
    import matplotlib
except ModuleNotFoundError:
    # This workspace already has matching CPython dependencies from previous lessons.
    sys.path.insert(0, str(WORK/'tmp/section3_1_deps'))
    import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':11, 'axes.unicode_minus':False,
                     'svg.fonttype':'none','figure.facecolor':'white',
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titlepad':12})

source = json.loads((OUT/'4.2-双势阱劈裂核验.json').read_text(encoding='utf-8'))
case = next(c for c in source['cases'] if c['a']==.4)
data = np.genfromtxt(OUT/'4.2-双势阱本征态与局域态.csv', delimiter=',', names=True)
x = data['x']; S=data['psi_even']; A=data['psi_odd']
R=data['psi_right']; L=data['psi_left']
ES,EA=case['even']['E'],case['odd']['E']; gap=EA-ES; Ebar=(ES+EA)/2
right=x>=0
q=float(np.trapezoid(S[right]*A[right],x[right]))
checks={}


def check(name,residual,tolerance=1e-10):
    error=float(np.max(np.abs(residual)))
    checks[name]={'max_abs_error':error,'tolerance':tolerance,'passed':error<=tolerance}
    if error>tolerance:
        raise AssertionError((name,error,tolerance))


check('even_parity',S-S[::-1]);check('odd_parity',A+A[::-1])
check('right_left_parity',R-L[::-1])
check('normalization',[np.trapezoid(f*f,x)-1 for f in [S,A,R,L]])
check('orthogonality',[np.trapezoid(S*A,x),np.trapezoid(R*L,x)])
check('right_construction',R-(S+A)/np.sqrt(2))
check('left_construction',L-(S-A)/np.sqrt(2))
sx=np.array([[0,1],[1,0]],complex);sz=np.diag([1.,-1.]);eye=np.eye(2)
H=Ebar*eye-gap/2*sx; parity=sx
even=np.array([1.,1.])/np.sqrt(2);odd=np.array([1.,-1.])/np.sqrt(2)
check('H_eigenstates',np.stack((H@even-ES*even,H@odd-EA*odd)))
check('parity_commutator',H@parity-parity@H)
check('parity_projectors',(eye+parity)@(eye-parity)/4)
theta=np.linspace(0,4*np.pi,401)
state=np.array([np.cos(theta/2),1j*np.sin(theta/2)]).T
derivative=np.array([-gap/2*np.sin(theta/2),1j*gap/2*np.cos(theta/2)]).T
check('reduced_Schrodinger_equation',1j*derivative-state@(H-Ebar*eye).T)
check('probability_sum',np.abs(state[:,0])**2+np.abs(state[:,1])**2-1)
even_prob=np.abs(state@even)**2;odd_prob=np.abs(state@odd)**2
check('parity_probabilities',np.stack((even_prob-.5,odd_prob-.5)))
check('parity_expectation',np.einsum('bi,ij,bj->b',state.conj(),parity,state))
check('initial_derivative',derivative[0]-np.array([0,1j*gap/2]))
errors=[];matrix_errors=[]
for th in np.linspace(0,4*np.pi,21):
    wave=(np.exp(1j*th/2)*S+np.exp(-1j*th/2)*A)/np.sqrt(2)
    actual=np.trapezoid(np.abs(wave[right])**2,x[right])
    errors.append(actual-(.5+q*np.cos(th)))
    t=th/gap
    spectral=(np.exp(-1j*ES*t)*even+np.exp(-1j*EA*t)*odd)/np.sqrt(2)
    local=np.exp(-1j*Ebar*t)*np.array([np.cos(th/2),1j*np.sin(th/2)])
    matrix_errors.extend(spectral-local)
check('half_space_interference_formula',errors)
check('full_phase_spectral_vs_local_basis',matrix_errors,tolerance=4e-10)

# Independent Gauss-Legendre quadrature for <2s|z|2p,m=0>, with a=1.
g,w=np.polynomial.legendre.leggauss(180)
r=30*(g+1);wr=30*w
R20=(2-r)*np.exp(-r/2)/(2*np.sqrt(2))
R21=r*np.exp(-r/2)/(2*np.sqrt(6))
radial=np.sum(wr*r**3*R20*R21)
angular=2*np.pi*np.sum(w*(1/np.sqrt(4*np.pi))*g*(np.sqrt(3/(4*np.pi))*g))
dipole=radial*angular
check('hydrogen_2s_2p_radial',radial+3*np.sqrt(3),tolerance=2e-11)
check('hydrogen_2s_2p_angular',angular-1/np.sqrt(3),tolerance=2e-12)
check('hydrogen_2s_2p_z',dipole+3,tolerance=2e-11)

blue,red,purple,green='#2369a1','#b54237','#74469c','#2a7f64'
fig,axs=plt.subplots(3,1,figsize=(10.8,9.1),sharex=True,
                     gridspec_kw={'height_ratios':[1,1.3,1.3]},layout='constrained')
fig.suptitle('对称矩形双阱：偶奇本征态与局域组合',fontsize=17,fontweight='bold')
axs[0].plot([-1.4,-.4,-.4,.4,.4,1.4],[0,0,100,100,0,0],color='#424b57',lw=2)
axs[0].fill_between(x,0,100,where=np.abs(x)<.4,color='#dce3eb',alpha=.7)
for wall in [-1.4,1.4]:
    axs[0].plot([wall,wall],[0,120],color='#424b57',lw=2.5)
    axs[0].text(wall,125,r'$\infty$',ha='center',va='bottom',fontsize=14)
axs[0].set_ylim(-9,144);axs[0].set_ylabel('势能 V')
axs[0].text(.5,.88,r'$\hbar=m=b=1,\quad a=0.4,\quad V_0=100$',transform=axs[0].transAxes,ha='center',
            bbox={'facecolor':'white','edgecolor':'none','alpha':.9},fontsize=11)
axs[1].plot(x,S,color=blue,lw=2.2,label=r'$S(x)$：偶宇称，最低态')
axs[1].plot(x,A,color=red,lw=2.0,ls='--',label=r'$A(x)$：奇宇称，次低态')
axs[1].set_ylabel('归一化波函数')
axs[1].legend(loc='upper center',ncols=2,frameon=False)
axs[1].text(.02,.06,f'$E_S={ES:.9f}$\n$E_A={EA:.9f}$',transform=axs[1].transAxes,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.9},fontsize=10)
axs[2].plot(x,R,color=green,lw=2.2,label=r'$R(x)=[S(x)+A(x)]/\sqrt{2}$')
axs[2].plot(x,L,color=purple,lw=2.0,ls='--',label=r'$L(x)=[S(x)-A(x)]/\sqrt{2}$')
axs[2].set_ylabel('归一化波函数');axs[2].set_xlabel('位置 x（长度单位 b）')
axs[2].legend(loc='upper center',ncols=2,frameon=False)
for ax in axs[1:]:
    ax.axhline(0,color='#8c98a5',lw=.8)
    ax.axvspan(-.4,.4,color='#dce3eb',alpha=.5,zorder=-1)
    ax.grid(alpha=.13)
axs[1].set_ylim(-1.16,1.32);axs[2].set_ylim(-.12,1.70)
axs[-1].set_xlim(-1.48,1.48)
axs[-1].set_xticks([-1.4,-1.,-.4,0,.4,1.,1.4])
fig.savefig(OUT/'4.2-双阱偶奇态与局域组合.png',dpi=165)
fig.savefig(OUT/'4.2-双阱偶奇态与局域组合.svg')
plt.close(fig)

fig,axs=plt.subplots(3,1,figsize=(10.8,9),sharex=True,layout='constrained',
                     gridspec_kw={'height_ratios':[1.4,.8,1]})
fig.suptitle('隧穿改变位置干涉，宇称概率保持不变',fontsize=17,fontweight='bold')
pr=np.cos(theta/2)**2;pl=np.sin(theta/2)**2
axs[0].plot(theta,pr,color=green,lw=2.2,label=r'$P_R^{\rm basis}=\cos^2(\theta/2)$')
axs[0].plot(theta,pl,color=purple,lw=2.2,ls='--',label=r'$P_L^{\rm basis}=\sin^2(\theta/2)$')
axs[0].plot(theta,.5+q*np.cos(theta),color='#3c434b',ls=':',lw=1.2,label=r'实际半空间 $P_{x>0}$')
axs[0].set_ylabel('局域基投影概率');axs[0].set_ylim(-.06,1.14)
axs[0].legend(loc='upper center',ncols=3,frameon=False,fontsize=10)
axs[0].annotate('首次到达 L',xy=(np.pi,1),xytext=(np.pi+.55,.72),
                arrowprops={'arrowstyle':'->','color':purple},color=purple)
axs[1].plot(theta,((.5+q*np.cos(theta))-pr)*1e8,color=red,lw=1.8)
axs[1].axhline(0,color='#808c99',lw=.8)
axs[1].set_ylabel(r'$[P_{x>0}-P_R^{\rm basis}]\times10^8$',fontsize=11)
axs[1].text(.02,.91,f'半空间与局域基的差异被放大；q={q:.11f}',
            transform=axs[1].transAxes,fontsize=10,va='top')
axs[1].set_ylim(-6.4,7.2)
axs[2].plot(theta,even_prob,color=blue,lw=2,label=r'偶宇称 $p_+=1/2$')
axs[2].plot(theta[::16],odd_prob[::16],ls='none',marker='x',ms=5,color=red,
            label=r'奇宇称 $p_-=1/2$')
axs[2].set_ylim(0,1);axs[2].set_ylabel('宇称测量概率')
axs[2].legend(loc='upper center',ncols=2,frameon=False)
axs[2].text(.5,.12,'S、A 分量的概率不变；相对相位在演化',ha='center',
            transform=axs[2].transAxes,fontsize=11)
axs[-1].set_xlabel(r'无量纲时间 $\theta=\Delta E\,t/\hbar$')
for ax in axs:
    ax.grid(alpha=.15);ax.set_xlim(0,4*np.pi)
axs[-1].set_xticks(np.arange(5)*np.pi,[r'$0$',r'$\pi$',r'$2\pi$',r'$3\pi$',r'$4\pi$'])
fig.savefig(OUT/'4.2-双阱隧穿概率与宇称守恒.png',dpi=165)
fig.savefig(OUT/'4.2-双阱隧穿概率与宇称守恒.svg')
plt.close(fig)

result={'status':'PASS','source_csv':'4.2-双势阱本征态与局域态.csv',
        'source_json':'4.2-双势阱劈裂核验.json','parameters':{'hbar':1,'m':1,'b':1,'a':.4,'V0':100},
        'ES':ES,'EA':EA,'gap':gap,'q':q,'initial_left_half_probability':.5-q,
        'first_R_to_L_time':np.pi/gap,'period':2*np.pi/gap,
        'hydrogen_z_matrix_element_in_units_of_a':float(dipole),'checks':checks,
        'scope':'Two-level model checks and finite-grid quadrature, not a proof of full continuum dynamics.'}
(OUT/'4.2-双阱图与两能级核验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'groups':len(checks),'q':q,
      'largest_residual':max(c['max_abs_error'] for c in checks.values())}))
