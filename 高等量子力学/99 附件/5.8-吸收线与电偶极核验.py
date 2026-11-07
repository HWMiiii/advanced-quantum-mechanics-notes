"""教学核验：洛伦兹谱线、氢1s→2p_x吸收、TRK与波长误差。"""
from pathlib import Path
import csv
import json
import os
import sys
out=Path(__file__).resolve().parent
root=out.parent.parent
deps=root/'tmp/section3_1_deps'
if sys.version_info[:2]==(3,12) and deps.is_dir():
    sys.path.insert(0,str(deps))
os.environ['MPLCONFIGDIR']=str(root/'tmp/section5_8_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists(): font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none','axes.spines.right':False,'axes.spines.top':False})
checks=[]
def check(name,computed,expected,tol):
    error=float(abs(computed-expected))
    checks.append({'name':name,'computed':float(computed),'expected':float(expected),'abs_error':error,'tolerance':tol,'passed':bool(error<tol)})
def line(x,gamma):
    return gamma/(2*np.pi)/(x*x+gamma*gamma/4)
def csv_data(name,columns,rows):
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(columns);writer.writerows(rows)
# Radial atomic units; the x-polarized real 2p_x orbital has x/r angular part.
r=np.linspace(0,50,25001)
R1=2*np.exp(-r)
R2=r*np.exp(-r/2)/(2*np.sqrt(6))
rad=np.trapezoid(R2*r**3*R1,r)
check('1s radial normalization',np.trapezoid(r*r*R1**2,r),1,1e-10)
check('2p radial normalization',np.trapezoid(r*r*R2**2,r),1,1e-10)
check('hydrogen radial dipole',rad,256/(81*np.sqrt(6)),1e-10)
angles,weights=np.polynomial.legendre.leggauss(48)
ang=np.dot(weights,angles**2)*np.sqrt(3)/2
check('dipole angular integral with polar axis along x',ang,1/np.sqrt(3),2e-14)
gap=3/8
x21=rad*ang
# -i d/dx on 1s gives i(x/r) exp(-r), same angular part, different radial moment.
p21=np.trapezoid(R2*r**2*R1,r)*ang
check('momentum length equivalence in atomic units',p21,gap*x21,2e-12)
strength=2*gap*x21*x21
check('hydrogen first-line oscillator strength',strength,8192/19683,3e-13)
for n in [0,1,3,8]:
    check(f'signed oscillator strengths from level {n}',(n+1)-n,1,1e-14)
for g in [.5,1,2]:
    check(f'Lorentzian FWHM gamma={g}',line(g/2,g)/line(0,g),.5,1e-13)
    # Integrate on a finite interval and restore the known analytic tail.
    x=np.linspace(-50,50,100001)
    tail=2*np.arctan(g/100)/np.pi
    check(f'Lorentzian area gamma={g}',np.trapezoid(line(x,g),x)+tail,1,5e-12)
z=np.linspace(-9,9,10001)
for s in [.01,.1,.3,1,2]:
    # Gaussian ground-state density in z. Compare the squared projected amplitude.
    amp=np.trapezoid(np.exp(-z*z+1j*s*z)/np.sqrt(np.pi),z)
    check(f'finite-k oscillator probability ratio k*b={s}',abs(amp)**2,np.exp(-s*s/2),3e-14)
alpha=1/137.035999177
unit_bohr_nm=0.0529177210544
values={'radial_in_a0':float(rad),'length_in_a0':float(x21),'f_2p_1s':float(strength),
        'k_a0':alpha*gap,'vacuum_wavelength_nm_fixed_nucleus':2*np.pi*unit_bohr_nm/(alpha*gap)}
phase=.61;h=1e-5
dtime=(np.cos(phase-h)-np.cos(phase+h))/(2*h)
check('E sign for cos(kz-omega t) in scaled units',-dtime,-np.sin(phase),3e-11)
x=np.linspace(-5,5,801);s=np.linspace(0,2.5,501)
csv_data('5.8-洛伦兹线型.csv',['detuning/gamma0','gamma0*L(width=gamma0)','gamma0*L(width=2gamma0)'],zip(x,line(x,1),line(x,2)))
csv_data('5.8-偶极近似的空间因子.csv',['k*b','squared_matrix_element_ratio'],zip(s,np.exp(-s*s/2)))
fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
a,b=axes
for g,col in [(1,'#007866'),(2,'#AA5027')]:
    a.plot(x,line(x,g),color=col,lw=2.35,label=rf'$\gamma={g}\gamma_0$')
    a.plot([-g/2,g/2],[line(0,g)/2]*2,color=col,lw=2,marker='|')
a.set(xlabel=r'失谐 $(\omega-\omega_{ni})/\gamma_0$',ylabel=r'归一线型 $\gamma_0 L_\gamma$',title='相同线面积，展宽降低峰值')
a.legend(frameon=False);b.plot(s,np.exp(-s*s/2),lw=2.5,color='#245A8A')
b.axhline(1,color='#777777',ls='--',lw=1.3,label='电偶极近似');b.legend(frameon=False)
for at in (.1,1,2):
    val=np.exp(-at*at/2);b.scatter([at],[val],s=20,color='#245A8A')
    b.annotate(f'{val:.3f}',(at,val),xytext=(4,-15 if at<.2 else 8),textcoords='offset points',fontsize=10)
b.set(xlabel=r'原子尺寸与光波数的乘积 $kb$',ylabel=r'$|M(k)|^2/|M(0)|^2$',title='同一振子跃迁的空间矩阵元',ylim=(0,1.13))
for ax in axes: ax.grid(axis='y',alpha=.18)
for ext in ('png','svg'): fig.savefig(out/f'5.8-谱线宽度与电偶极近似.{ext}',dpi=180)
plt.close(fig)
(out/'5.8-吸收线与电偶极核验.json').write_text(json.dumps({'model':'single-electron local potential; illustrative Lorentzian and oscillator models','values':values,'checks':checks},ensure_ascii=False,indent=2),encoding='utf8')
assert all(item['passed'] for item in checks),[c for c in checks if not c['passed']]
print(len(checks),values)

