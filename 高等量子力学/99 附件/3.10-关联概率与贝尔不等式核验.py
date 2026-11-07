"""Supplementary spin-singlet/Bell checks and reproducible figures.

Basis: |++>, |+->, |-+>, |--> in the z basis. hbar=1.
Outcomes r,s are +1/-1 for sigma, so physical spin results are r/2,s/2.
Requires numpy and matplotlib. Outputs are written beside this script.
No source textbook or other notes are modified.
"""
from pathlib import Path
from itertools import product
from math import pi, sqrt
import json
import numpy as np

OUT = Path(__file__).resolve().parent
TOL = 5e-13
I = np.eye(2, dtype=complex)
PAULI = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]], dtype=complex)
PSI = np.array([0,1,-1,0], dtype=complex)/sqrt(2)
RHO = np.outer(PSI,PSI.conj())
MIX = np.diag([0,.5,.5,0]).astype(complex)


def sigma(n):
    return np.tensordot(np.asarray(n), PAULI, axes=1)


def axis(theta):
    return np.array([np.sin(theta), 0, np.cos(theta)])


def projector(n, outcome):
    return (I+outcome*sigma(n))/2


def trace_b(rho):
    return np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)


def trace_a(rho):
    return np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)


def joint(rho, a, b, r, s):
    return float(np.trace(rho@np.kron(projector(a,r),projector(b,s))).real)


def correlation(rho,a,b):
    return float(np.trace(rho@np.kron(sigma(a),sigma(b))).real)


def run_checks():
    counts, maxima = {}, {}
    def check(name, difference):
        err=float(np.max(np.abs(difference)))
        counts[name]=counts.get(name,0)+1
        maxima[name]=max(maxima.get(name,0),err)
        if err>TOL:
            raise AssertionError(f'{name}: {err} exceeds {TOL}')

    directions=[np.array(x,dtype=float) for x in [(0,0,1),(1,0,0),(0,1,0),(0,0,-1)]]
    directions += [axis(t) for t in [-pi/3,-pi/4,pi/6,pi/4,pi/3,2*pi/3]]
    directions += [v/np.linalg.norm(v) for v in map(np.array,[(1,2,3),(-2,1,4),(3,-2,1)])]
    check('singlet_purity',RHO@RHO-RHO)
    check('singlet_reduced_A',trace_b(RHO)-I/2)
    check('singlet_reduced_B',trace_a(RHO)-I/2)
    check('mixture_purity',np.trace(MIX@MIX)-.5)
    check('mixture_reduced_A',trace_b(MIX)-I/2)
    for a in directions:
        for b in directions:
            probabilities=[]
            for r,s in product([1,-1],repeat=2):
                p=joint(RHO,a,b,r,s)
                probabilities.append(p)
                check('joint_probability',p-(1-r*s*np.dot(a,b))/4)
            check('joint_normalization',sum(probabilities)-1)
            for r in [1,-1]:
                check('marginal_A',sum(joint(RHO,a,b,r,s) for s in [1,-1])-.5)
            for s in [1,-1]:
                check('marginal_B',sum(joint(RHO,a,b,r,s) for r in [1,-1])-.5)
            check('singlet_correlation',correlation(RHO,a,b)+np.dot(a,b))
            check('mixture_correlation',correlation(MIX,a,b)+a[2]*b[2])
        nonselective=np.zeros((4,4),complex)
        for s in [1,-1]:
            kb=np.kron(I,projector(a,s))
            post=kb@RHO@kb
            nonselective+=post
            prob=np.trace(post).real
            check('selection_probability',prob-.5)
            check('conditional_state_A',trace_b(post)/prob-(I-s*sigma(a))/2)
        check('nonselective_A_unchanged',trace_b(nonselective)-I/2)
        check('nonselective_B_unchanged',trace_a(nonselective)-I/2)
        check('nonselective_global_purity',np.trace(nonselective@nonselective)-.5)
    z,x=axis(0),axis(pi/2)
    check('z_dephased_equals_classical_mixture',sum(np.kron(I,projector(z,s))@RHO@np.kron(I,projector(z,s)) for s in [1,-1])-MIX)
    check('singlet_xx',correlation(RHO,x,x)+1)
    check('mixture_xx',correlation(MIX,x,x))

    wigner_rows=[]
    for signs in product([1,-1],repeat=3):
        a,b,c=signs
        lhs=int(a==1 and -b==1)
        ac=int(a==1 and -c==1)
        cb=int(c==1 and -b==1)
        slack=ac+cb-lhs
        if slack<0: raise AssertionError('Local Wigner violation')
        check('wigner_deterministic_bound',min(slack,0))
        wigner_rows.append({'A_a_b_c':list(signs),'B_a_b_c':[-v for v in signs], 'Pab_indicator':lhs,'Pac_indicator':ac,'Pcb_indicator':cb,'R_minus_L':slack})
    chsh_values=[]
    for a,ap,b,bp in product([1,-1],repeat=4):
        s=a*(b+bp)+ap*(b-bp)
        check('CHSH_deterministic_bound',abs(s)-2)
        chsh_values.append({'A':a,'Aprime':ap,'B':b,'Bprime':bp,'S':s})
    a,ap,b,bp=z,x,axis(pi/4),axis(-pi/4)
    schsh=correlation(RHO,a,b)+correlation(RHO,a,bp)+correlation(RHO,ap,b)-correlation(RHO,ap,bp)
    check('CHSH_quantum_value',schsh+2*sqrt(2))
    bop=np.kron(sigma(a),sigma(b)+sigma(bp))+np.kron(sigma(ap),sigma(b)-sigma(bp))
    check('CHSH_operator_norm',np.linalg.norm(bop,2)-2*sqrt(2))
    check('CHSH_singlet_expectation',np.trace(RHO@bop).real-schsh)

    bell_examples=[]
    for theta in [0,pi/6,pi/4,pi/3,pi/2]:
        a,b,c=axis(-theta),axis(theta),z
        lhs=joint(RHO,a,b,1,1)
        rhs=joint(RHO,a,c,1,1)+joint(RHO,c,b,1,1)
        check('Wigner_symmetric_L',lhs-.5*np.sin(theta)**2)
        check('Wigner_symmetric_R',rhs-np.sin(theta/2)**2)
        check('Wigner_violation_formula',lhs-rhs-.5*np.cos(theta)*(1-np.cos(theta)))
        bell_examples.append({'theta_radian':theta,'theta_over_pi':theta/pi,'L_joint':lhs,'R_joint':rhs,'L_minus_R':lhs-rhs,'twice_L_book':2*lhs,'twice_R_book':2*rhs})
    check('max_Wigner_violation',bell_examples[3]['L_minus_R']-.125)
    result={'status':'PASS','hbar':1,'tolerance':TOL,'basis':['++','+-','-+','--'],
        'directions': [n.tolist() for n in directions], 'direction_pairs':len(directions)**2,
        'check_counts':counts,'maximum_absolute_errors':maxima,'total_checks':sum(counts.values()),
        'Wigner_8_types':wigner_rows,'CHSH_16_types':chsh_values,'CHSH_quantum_S':schsh,
        'Wigner_symmetric_examples':bell_examples,
        'scope':'Finite deterministic algebra/matrix checks; no experimental data or Monte Carlo significance claims.'}
    (OUT/'3.10-关联概率与贝尔不等式核验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def make_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fp=Path('C:/Windows/Fonts/msyh.ttc')
    if fp.exists(): matplotlib.rcParams['font.family']=FontProperties(fname=fp).get_name()
    matplotlib.rcParams['axes.unicode_minus']=False
    matplotlib.rcParams['svg.fonttype']='none'
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    fig.suptitle('自旋单态：条件概率、联合概率与贝尔不等式（教学补充）',fontsize=15)
    t=np.linspace(0,pi,501)
    ax.plot(t/pi,.5*np.sin(t/2)**2,lw=2.5,color='#2966a2',label=r'联合 $P(+,+\mid a,b)$')
    ax.plot(t/pi,np.sin(t/2)**2,lw=2.5,color='#c06a2e',label=r'条件 $P(B+\mid A+,a,b)$')
    ax.plot(t/pi,np.full_like(t,.5),lw=1.6,ls='--',color='#72777d',label=r'边缘 $P(B+\mid b)$')
    ax.scatter([.5,.5],[.25,.5],color=['#2966a2','#c06a2e'],s=35,zorder=5)
    ax.text(.53,.21,r'$1/4$',fontsize=12,color='#2966a2')
    ax.text(.53,.54,r'$1/2$',fontsize=12,color='#c06a2e')
    ax.set(xlabel=r'两测量轴夹角 $\theta_{ab}/\pi$',ylabel='概率',xlim=(0,1),ylim=(-.02,1.03),xticks=[0,.25,.5,.75,1])
    ax.set_title('A  同一条件事件不能省掉其发生概率',fontsize=12)
    ax.legend(loc='upper left',fontsize=10,framealpha=.95)
    ax.grid(alpha=.2)

    theta=np.linspace(0,pi/2,501)
    l=.5*np.sin(theta)**2
    r=np.sin(theta/2)**2
    bx.plot(theta/pi,l,lw=2.5,color='#2966a2',label=r'$L=P(a+,b+)=\frac{1}{2}\sin^2\theta$')
    bx.plot(theta/pi,r,lw=2.5,color='#c06a2e',label=r'$R=P(a+,c+)+P(c+,b+)=\sin^2(\theta/2)$')
    bx.fill_between(theta/pi,r,l,color='#b899cc',alpha=.3,label=r'量子违反区：$L>R$')
    bx.scatter([.25,.25],[.25,(2-sqrt(2))/4],color=['#2966a2','#c06a2e'],s=35,zorder=5)
    bx.plot([1/3,1/3],[.25,.375],color='#604283',lw=2,ls=':')
    bx.text(.29,.45,r'$\Delta_{\max}=1/8$'+'\n'+r'$\theta=\pi/3$',fontsize=11,color='#604283',ha='left',bbox={'facecolor':'white','edgecolor':'#ccc','alpha':.95})
    bx.text(.06,.38,r'$\theta=\pi/4$'+'\n'+r'$L=0.2500$'+'\n'+r'$R=0.1464$',fontsize=11,bbox={'facecolor':'white','edgecolor':'#ccc','alpha':.95})
    bx.set(xlabel=r'对称设置中的半夹角 $\theta/\pi$',ylabel='联合概率或其和',xlim=(0,.5),ylim=(-.01,.53),xticks=[0,1/6,.25,1/3,.5],xticklabels=['0','1/6','1/4','1/3','1/2'])
    bx.set_title(r'B  $\theta_{ab}=2\theta,\ \theta_{ac}=\theta_{cb}=\theta$',fontsize=12)
    bx.legend(loc='lower right',fontsize=8.8,framealpha=.95)
    bx.grid(alpha=.2)
    for ext in ['png','svg']: fig.savefig(OUT/f'3.10-条件概率与贝尔违反.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    result=run_checks()
    make_figure()
    print(json.dumps({'status':result['status'],'total_checks':result['total_checks'],'maximum_error':max(result['maximum_absolute_errors'].values()),'CHSH_S':result['CHSH_quantum_S']},ensure_ascii=False))
