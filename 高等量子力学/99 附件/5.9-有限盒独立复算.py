"""给5.51的连续谱结果作独立检查。用远处第二面硬墙的离散本征态演化。
    只比较有限时间内的井内量；有限盒不用于证明无限域的晚期幂律。
"""
from pathlib import Path
import csv
import json
import os
import sys
root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
deps = root/'tmp/section3_1_deps'
if sys.version_info[:2] == (3,12) and deps.is_dir():
    sys.path.insert(0, str(deps))
os.environ['MPLCONFIGDIR'] = str(root/'tmp/section5_9_mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.is_file(): font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':11,
                     'axes.spines.right':False,'axes.spines.top':False,
                     'svg.fonttype':'none','axes.unicode_minus':False})
G = 6.
def phase(q):
    s = np.sin(q)
    return np.arctan2(-G*s*s/q, 1+G*s*np.cos(q)/q)
def solve(B, cutoff, times, x_nodes=72):
    # y=(x+a)/a. Walls at y=0, B. Outside well, wave = R sin(qy+phase).
    j = np.arange(1,int(B*cutoff/np.pi))
    lo = j*np.pi/B
    hi = (j+1)*np.pi/B
    for _ in range(39):
        mid = (lo+hi)/2
        high = B*mid+phase(mid)>j*np.pi
        lo = np.where(high,lo,mid)
        hi = np.where(high,mid,hi)
    q=(lo+hi)/2; delta=phase(q)
    sqr_R = (1+G*np.sin(q)*np.cos(q)/q)**2+(G*np.sin(q)**2/q)**2
    norm_sq = .5-np.sin(2*q)/(4*q)+sqr_R*((B-1)/2+np.sin(2*q+2*delta)/(4*q))
    # initial sqrt(2) sin(pi(y-1)), zero outside; np.sinc(x)=sin(pi*x)/(pi*x).
    projection = -1/np.sqrt(2)*(np.sinc((q-np.pi)/np.pi)-np.sinc((q+np.pi)/np.pi))
    overlaps_sq = projection**2/norm_sq
    y,w = np.polynomial.legendre.leggauss(x_nodes)
    y=(y+1)/2; w=w/2
    interior = (projection/norm_sq)[:,None]*np.sin(q[:,None]*y)
    evolve = np.exp(-1j*np.outer(times,q*q))
    psi = evolve@interior
    P = (abs(psi)**2)@w
    P0 = abs(evolve@overlaps_sq)**2
    return P,P0,float(sum(overlaps_sq)),float(np.max(abs(np.sin(B*q+delta))))
times = np.array([0,.1,.5,1,2,4,6,8,12])
runs={}
for B,cut in [(900.37,55),(1350.29,75),(1500.47,95)]:
    P,P0,weight,boundary = solve(B,cut,times)
    runs[f'{B}_{cut}']={'far_wall':B,'max_wave_number':cut,'overlap_weight':weight,
                         'wall_residual':boundary,'prob_in_well':P.tolist(),'return_prob':P0.tolist()}
final=list(runs.values())[-1]; prev=list(runs.values())[-2]; coarse=list(runs.values())[0]
checks=[]
def check(name,err,tol):
    checks.append({'name':name,'max_abs_error':float(err),'tolerance':tol,'passed':bool(err<tol)})
for tag,run in runs.items():
    check('normalization '+tag,abs(run['overlap_weight']-1),3e-5)
    check('distant wall '+tag,run['wall_residual'],2e-10)
    check('projection inequality '+tag,max(0.,np.max(np.array(run['return_prob'])-run['prob_in_well'])),1e-12)
for index,t in enumerate(times[1:],1):
    p=final['prob_in_well'][index]
    check(f'refinement at T={t:g}',abs(prev['prob_in_well'][index]-p)/p,3e-4)
# One-parameter prediction from matching an outgoing exterior, independent of the fit.
q=2.75-.14j
for _ in range(15):
    f=q/np.tan(q)+G-1j*q
    df=1/np.tan(q)-q/np.sin(q)**2-1j
    q-=f/df
tau=-1/(2*(q*q).imag)
check('outgoing root',abs(q/np.tan(q)+G-1j*q),1e-12)
# Physical-unit illustration. Width of an upper level from the 5.8 hydrogen lifetime.
hbar_eVs=6.582119569e-16
tau_H=1.59533e-9
box_data={'units':'y=(x+a)/a, energy unit hbar^2/(2ma^2), G=6,n=1','T':times.tolist(),'runs':runs,
          'pole_q':[q.real,q.imag], 'pole_q_squared':[(q*q).real,(q*q).imag],'pole_tau_dimensionless':tau,
          'illustration_2p_width_eV':hbar_eVs/tau_H,'checks':checks}
(out/'5.9-有限盒独立复算核验.json').write_text(json.dumps(box_data,ensure_ascii=False,indent=2),encoding='utf8')
with (out/'5.9-有限盒两种概率.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['T','P_in_well','P_return','B','q_max'])
    writer.writerows(zip(times,final['prob_in_well'],final['return_prob'],[final['far_wall']]*len(times),[final['max_wave_number']]*len(times)))
# Show the same energy scale in time and energy; independent teaching picture.
fig,(a,b)=plt.subplots(1,2,figsize=(10.3,3.85),layout='constrained')
t=np.linspace(0,4,401)
a.plot(t,np.exp(-t),color='#007568',lw=2.6,label='留在初态的概率')
a.plot(t,np.exp(-t/2),color='#95612f',ls='--',lw=2,label='初态振幅的模')
a.plot([1],[np.exp(-1)],'o',color='#007568')
a.set(xlabel=r'时间 $t/\tau$',ylabel='相对初值',title='同一条衰变曲线的两种读数')
a.legend(frameon=False)
x=np.linspace(-2,2,601);b.plot(x,1/(1+4*x*x),color='#285989',lw=2.5)
b.plot([-.5,.5],[.5,.5],color='#a75a2a',marker='|',lw=2)
b.text(0,.42,'半高全宽 = Γ',ha='center',color='#92451f')
b.set(xlabel=r'相对峰心的能量 $(E-E_{\rm c})/\Gamma$',ylabel='相对峰值',title='能量分布的全宽')
for ax in (a,b): ax.grid(axis='y',alpha=.15)
fig.savefig(out/'5.9-寿命与能量全宽.png',dpi=165)
plt.close(fig)
print('pole',q,'tau',tau,'checks',len(checks),'passed',all(c['passed'] for c in checks))
for t,p,p0 in zip(times,final['prob_in_well'],final['return_prob']): print(t,p,p0)
if not all(c['passed'] for c in checks): raise ValueError([c for c in checks if not c['passed']])

