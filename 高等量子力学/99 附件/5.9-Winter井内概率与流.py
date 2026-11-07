"""Sakurai 5.51(b,c), n=1, G=6. Only NumPy and Matplotlib are needed.
The principal calculation rotates the q contour and includes crossed poles.
Cross-checks use an independently integrated contour above every pole, the
original real-axis integral, continuity, refined quadratures and the late-time law.
All CSV and plotted times are T=hbar*t/(2*m*a^2), and J=-dP/dT.
"""
from functools import lru_cache
import csv
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent.parent
deps = PROJECT/'tmp'/'section3_1_deps'
if sys.version_info[:2] == (3, 12) and deps.is_dir():
    sys.path.insert(0, str(deps))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT/'tmp'/'section5_9_winter'/'mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

N=1
G=6.
PI=np.pi
DIRECTION=np.exp(-1j*PI/4)
C_TAIL=1/(6*N*N*PI**3*(1+G)**4)


@lru_cache(None)
def gauss(n):
    return np.polynomial.legendre.leggauss(n)


def mapped_gauss(n, start, stop):
    nodes, weights = gauss(n)
    return (start+stop)/2+(stop-start)*nodes/2, (stop-start)*weights/2


def factors(q):
    # Stable near q=0; F+ has outgoing resonance roots in the fourth quadrant.
    return q+G*np.expm1(2j*q)/(2j), q-G*np.expm1(-2j*q)/(2j)


def coefficient(q):
    fplus,fminus=factors(q)
    # sin q/(q^2-n^2*pi^2) has no pole at n*pi. np.sinc(z)=sin(pi*z)/(pi*z).
    regular=(-1)**N*np.sinc((q-N*PI)/PI)/(q+N*PI)
    return 2*N*np.sqrt(2)*q*q*regular/(fplus*fminus)


@lru_cache(None)
def poles(max_count):
    k=np.arange(1,max_count+1)
    q=k*PI-.4-.2j
    for _ in range(35):
        q=k*PI+np.log(1-2j*q/G)/(2j)
    fplus,fminus=factors(q)
    if np.max(np.abs(fplus)/(1+np.abs(q)))>5e-12:
        raise ValueError('Outgoing resonance roots did not converge')
    derivative=1+G*np.exp(2j*q)
    residues=2*N*np.sqrt(2)*q*q*np.sin(q)/((q*q-(N*PI)**2)*derivative*fminus)
    return q,residues


def integrate_terms(q, weights, y):
    """Return chi at near-well y nodes, plus its spatial derivative at y=1."""
    phi=np.sum(weights[:,None]*np.sin(q[:,None]*y[None,:]),axis=0)
    derivative=np.sum(weights*q*np.cos(q))
    return phi,derivative


def rotated_wave(T,y,order=144,zmax=9.,damping=46.):
    if T==0:
        return np.sqrt(2)*np.sin(N*PI*(y-1)),np.sqrt(2)*N*PI
    z,wz=mapped_gauss(order,0,zmax)
    q=DIRECTION*z/np.sqrt(T)
    rayweights=DIRECTION*wz/np.sqrt(T)*np.exp(-z*z)*coefficient(q)
    wave,derivative=integrate_terms(q,rayweights,y)
    q,res=poles(1800)
    if -T*np.imag(q[-1]**2)<damping:
        raise ValueError('Increase resonance list at this very short time')
    keep=(-T*np.imag(q*q)<damping)
    q=q[keep]
    poleweights=-2j*PI*res[keep]*np.exp(-1j*T*q*q)
    p_wave,p_derivative=integrate_terms(q,poleweights,y)
    return wave+p_wave,derivative+p_derivative


def observables(T,spatial_order=64,ray_order=144,zmax=9.,damping=46.):
    y,wy=mapped_gauss(spatial_order,0,1)
    wave,derivative=rotated_wave(T,np.append(y,1.),ray_order,zmax,damping)
    p=float(np.dot(wy,np.abs(wave[:-1])**2))
    j=float(2*np.imag(np.conj(wave[-1])*derivative))
    return p,j


def horizontal_wave(T,y,offset=.08,real_cut=None):
    """Without residues: descend before first pole, then integrate horizontally.
    offset=0 calculates the unmodified real integral to a specified real_cut.
    Limit panel length both in q and in oscillatory phase, with 12-point GL/panel.
    """
    if offset:
        q0,dq0=mapped_gauss(48,0,-1j*offset)
        wave,derivative=integrate_terms(q0,dq0*coefficient(q0)*np.exp(-1j*T*q0*q0),y)
        cutoff=30/(2*offset*T) if real_cut is None else real_cut
    else:
        wave=np.zeros(len(y),complex);derivative=0j
        cutoff=real_cut
    phase_steps=np.sqrt(2*PI*np.arange(1,int(T*cutoff**2/(2*PI))+1)/T)
    regular_steps=np.arange(0,min(18.,cutoff),.11)
    edges=np.unique(np.concatenate([[0.,cutoff],phase_steps,regular_steps]))
    nodes,weights=gauss(12)
    for first in range(0,len(edges)-1,256):
        left=edges[first:first+256]
        right=edges[first+1:first+257]
        left=left[:len(right)]
        half=(right-left)/2
        q=((right+left)[:,None]/2+half[:,None]*nodes-1j*offset).ravel()
        dq=(half[:,None]*weights).ravel()
        term,deriv=integrate_terms(q,dq*coefficient(q)*np.exp(-1j*T*q*q),y)
        wave+=term;derivative+=deriv
    return wave,derivative


def fit_decay(times,probabilities):
    slope,intercept=np.polyfit(times,np.log(probabilities),1)
    fitted=np.exp(intercept+slope*times)
    errors=probabilities/fitted-1
    return {'tau':float(-1/slope),'slope':float(slope),'C':float(np.exp(intercept)),
            'max_fractional_residual':float(np.max(np.abs(errors))),
            'rms_fractional_residual':float(np.sqrt(np.mean(errors**2)))}


def fit_unweighted(times,probabilities):
    def err(tau):
        f=np.exp(-times/tau);amp=np.dot(f,probabilities)/np.dot(f,f)
        return np.sum((amp*f-probabilities)**2)
    lo,hi=.6,.7
    for _ in range(65):
        a=lo+(hi-lo)/3; b=hi-(hi-lo)/3
        if err(a)<err(b): hi=b
        else: lo=a
    return (lo+hi)/2


def zero_crossing(a,b):
    ja=observables(a)[1]
    for _ in range(35):
        mid=(a+b)/2
        jm=observables(mid)[1]
        if jm*ja>0: a=mid;ja=jm
        else: b=mid
    return (a+b)/2


def make_figures(rows,fit):
    for font in ('C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf'):
        if Path(font).is_file():
            font_manager.fontManager.addfont(font)
            plt.rcParams['font.family']=[font_manager.FontProperties(fname=font).get_name(),'DejaVu Sans']
            break
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
    T,P,J=rows.T[:3]
    fig,axes=plt.subplots(1,2,figsize=(11.,4.2),layout='constrained')
    inrange=T<=12
    axes[0].semilogy(T[inrange],P[inrange],lw=2,color='#225f80',label='井内概率 P(T)')
    tf=np.linspace(2,8,100)
    axes[0].semilogy(tf,fit['C']*np.exp(-tf/fit['tau']),ls='--',color='#da7942',lw=1.8,label='2 ≤ T ≤ 8 指数拟合')
    axes[0].set(xlim=(0,12),xlabel='无量纲时间 T',ylabel='井内概率',title='原题 (b)　固定区间内的总概率')
    axes[0].legend(frameon=False)
    zoom=(T>=1)&(T<=12)
    relative=P[zoom]/(fit['C']*np.exp(-T[zoom]/fit['tau']))-1
    axes[1].plot(T[zoom],100*relative,color='#00785e',lw=1.7)
    axes[1].axvspan(2,8,color='#dce8e5',alpha=.65,zorder=0);axes[1].axhline(0,color='#777',lw=.7)
    axes[1].set(xlim=(1,12),xlabel='无量纲时间 T',ylabel='相对拟合曲线的偏差（%）',title='偏差随时间增长并出现振荡')
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(OUT/'5.9-Winter井内概率与指数拟合.png',dpi=190,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.,4.2),layout='constrained')
    early=(T>=0)&(T<=24)
    axes[0].plot(T[early],J[early],color='#8b382f',lw=1.15,label='出口净流 J')
    axes[0].axhline(0,color='#687077',lw=.7)
    axes[0].set_yscale('symlog',linthresh=1e-12,linscale=.6)
    axes[0].set(xlim=(0,24),ylim=(-2e-7,3.),xlabel='无量纲时间 T',ylabel='有符号的概率流 J',title='原题 (c)　负的净流意味着井内概率回升')
    axes[0].set_yticks([-1e-7,-1e-9,-1e-11,0,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1])
    axes[0].legend(frameon=False,loc='upper right')
    tail=(T>=8)
    axes[1].loglog(T[tail],P[tail],color='#225f80',lw=1.9,label='井内概率 P')
    axes[1].loglog(T[tail],C_TAIL/T[tail]**3,color='#da7942',ls='--',label='阈值渐近 C / T³')
    axes[1].loglog(T[tail],3*C_TAIL/T[tail]**4,color='#48806a',ls=':',label='净流渐近 3C / T⁴')
    positive=(T>=30)&(J>0)
    axes[1].loglog(T[positive],J[positive],color='#116553',lw=1.4,label='净流 J（T ≥ 30）')
    axes[1].set(xlabel='无量纲时间 T（对数轴）',ylabel='井内概率与无量纲流',title='长时间的幂律尾部')
    axes[1].legend(frameon=False,fontsize=8.3)
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(OUT/'5.9-Winter出口流与晚期尾部.png',dpi=190,bbox_inches='tight');plt.close(fig)


def main():
    times=np.unique(np.round(np.concatenate([np.arange(0,1.005,.005),np.arange(1,24.005,.01),
                         np.arange(24,40.05,.05),np.geomspace(40,256,81),[50,64,80,100,128,160,192,256]]),9))
    times=times[times<=256]
    rows=[]
    for t in times:
        p,j=observables(t)
        rows.append([t,p,j,j/p])
    rows=np.array(rows)
    # Fit uniform T grid (2 to 8 inclusive) independently of merged plot sampling.
    tf=np.linspace(2,8,121)
    pf=np.array([observables(t)[0] for t in tf])
    fit=fit_decay(tf,pf)
    fit['sampling']='121 equally spaced samples, dT=0.05 inclusive'
    fit['unweighted_probability_space_tau']=float(fit_unweighted(tf,pf))
    fit['delta_vs_winter_percent']=float(100*(fit['tau']/.644-1))
    fit['fit_every_other_sample']=fit_decay(tf[::2],pf[::2])['tau']
    q1=poles(1800)[0][0]
    fit['pole']=[q1.real,q1.imag]
    fit['pole_energy']=[(q1*q1).real,(q1*q1).imag]
    fit['tau_from_first_pole']=float(-1/(2*(q1*q1).imag))
    signs=[]
    for i in range(1,len(rows)):
        if rows[i-1,2]*rows[i,2]<0:
            signs.append(zero_crossing(rows[i-1,0],rows[i,0]))
    intervals=[]
    for beg,end in zip(signs[::2],signs[1::2]):
        p1=observables(beg)[0];p2=observables(end)[0]
        intervals.append({'start':beg,'end':end,'P_at_start':p1,'P_at_end':p2,
                          'relative_rise':p2/p1-1})
    diagnostics=[]
    def check(name,residual,tolerance):
        residual=float(abs(residual))
        diagnostics.append({'name':name,'residual':residual,'tolerance':tolerance,'passed':residual<tolerance})
    check('P(0)=1',rows[0,1]-1,2e-13);check('J(0)=0',rows[0,2],1e-15)
    check('no-pole contour below real axis but above first resonance',max(0,.08+q1.imag),1e-15)
    for t in [.005,.04,.3,2.,8.,11.8,14.,25.,80.,256.]:
        p,j=observables(t)
        p2,j2=observables(t,96,200,10.,55.)
        check(f'收敛 P T={t}',abs(p2-p)/p,8e-7)
        check(f'收敛 J T={t}',abs(j2-j)/max(abs(j),p/t),1e-5)
    for t in [.5,2.,8.,13.12,16.,28.,80.]:
        y,wy=mapped_gauss(36,0,1)
        vals,d=horizontal_wave(t,np.append(y,1.))
        ptest=float(np.dot(wy,np.abs(vals[:-1])**2))
        jtest=float(2*np.imag(np.conj(vals[-1])*d))
        p,j=observables(t)
        check(f'无极点独立路径 P T={t}',abs(p-ptest)/p,2e-8)
        check(f'无极点独立路径 J T={t}',abs(j-jtest)/max(abs(j),p/t),5e-8)
    real_axis_checks=[]
    for t in [1.,5.]:
        y=np.array([.36,.73,1.])
        target,_=rotated_wave(t,y)
        errs=[]
        for cut in [80.,160.,320.]:
            estimate,_=horizontal_wave(t,y,offset=0.,real_cut=cut)
            errs.append(float(np.max(np.abs(estimate-target))))
        real_axis_checks.append({'time':t,'q_cuts':[80.,160.,320.],'max_wave_abs_errors':errs})
        check(f'原始实轴截断波函数 T={t}',errs[-1],3e-7)
    for t in [.3,2.,8.,13.12,16.,28.,80.,256.]:
        p,j=observables(t)
        h=.0001 if t<20 else .001
        derivative=(observables(t-2*h)[0]-8*observables(t-h)[0]+8*observables(t+h)[0]-observables(t+2*h)[0])/(12*h)
        check(f'独立五点差分连续性 T={t}',abs(j+derivative)/max(abs(j),p/t),3e-7)
    check('晚期P的解析系数 T=256',rows[-1,1]*256**3/C_TAIL-1,2e-5)
    check('晚期J的解析系数 T=256',rows[-1,2]*256**4/(3*C_TAIL)-1,4e-5)
    check('回流区间包含上升的P',float(any(r['relative_rise']<=0 for r in intervals)),1e-14)
    # Winter p. 1505 checked this specific window on both sides of the first dip.
    ta,tb=10.75,10.95
    times_dip,weights_dip=mapped_gauss(32,ta,tb)
    p_before,p_after=observables(ta)[0],observables(tb)[0]
    net_flux=float(weights_dip@np.array([observables(t)[1] for t in times_dip]))
    winter_window={'T_start':ta,'T_end':tb,'P_start':p_before,'P_end':p_after,
                   'P_increase':p_after-p_before,'integrated_outward_flow':net_flux}
    check('Winter原文首谷区间的积分连续性',(p_after-p_before+net_flux)/(p_after-p_before),1e-9)
    with (OUT/'5.9-Winter概率与出口流.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.writer(fp);writer.writerow(['T','P_inside','J_at_zero','J_div_P']);writer.writerows(rows)
    with (OUT/'5.9-Winter指数拟合.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.writer(fp);writer.writerow(['T','P_inside','P_exponential','relative_residual'])
        for t,p in zip(tf,pf):
            fitval=fit['C']*np.exp(-t/fit['tau']);writer.writerow([t,p,fitval,p/fitval-1])
    sample_times=[0,.05,.5,2,4,8,10,12,14,16,20,24,30,40,64,128,256]
    report={'source':'教材PDF385—386／印刷363—364；Winter, Phys. Rev. 123,1503 (1961), doi:10.1103/PhysRev.123.1503',
            'n':N,'G':G,'C_tail':C_TAIL,'fit':fit,'negative_current_intervals':intervals,
            'samples':[[t,*observables(t)] for t in sample_times],'total_csv_times':len(times),
            'direct_real_axis':real_axis_checks,'winter_window':winter_window,
            'checks':diagnostics,'check_count':len(diagnostics),
            'passed':all(c['passed'] for c in diagnostics)}
    (OUT/'5.9-Winter数值核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'fit':fit,'negative':intervals,'samples':report['samples'],'real_axis':real_axis_checks,
                      'passed':report['passed'],'failed':[c for c in diagnostics if not c['passed']]},ensure_ascii=False))
    if not report['passed']:
        raise RuntimeError('Numerical audit did not pass; see JSON.')
    make_figures(rows,fit)


if __name__=='__main__':
    main()
