"""§4.3 教学补充：N=8周期环、最近邻能带和构造的Bloch波形。

有限环: T|n>=|n+1 mod N>, H=E0 I-Delta(T+T†), E0=0,Delta=1,N=8。
Bloch波形仅演示变换与傅里叶分量，不是某个指定连续势的本征解。
所有图、CSV和JSON均为本脚本实际计算输出。
"""
from pathlib import Path
import csv, json, os, sys
OUT=Path(__file__).resolve().parent
WORK=OUT.parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(WORK/'tmp/pdfs/section4_3/mpl'))
try:
    import matplotlib
except ModuleNotFoundError:
    sys.path.insert(0,str(WORK/'tmp/section3_1_deps'))
    import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none',
                    'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
checks={}


def check(name,residual,tolerance=2e-12):
    err=float(np.max(np.abs(residual)))
    checks[name]={'max_abs_error':err,'tolerance':tolerance,'passed':err<=tolerance}
    if err>tolerance: raise AssertionError((name,err,tolerance))


N=8;E0=0.;Delta=1.;a=1.;hbar=1.
indices=np.arange(N)
angles=2*np.pi*indices/N
folded=(angles+np.pi)%(2*np.pi)-np.pi
T=np.zeros((N,N),complex)
for n in range(N):T[(n+1)%N,n]=1
H=E0*np.eye(N)-Delta*(T+T.conj().T)
F=np.exp(1j*np.outer(indices,angles))/np.sqrt(N)
phases=np.exp(-1j*angles)
energies=E0-2*Delta*np.cos(angles)
energies[np.abs(energies)<1e-14]=0
check('T_unitarity',T.conj().T@T-np.eye(N))
check('T_to_N',np.linalg.matrix_power(T,N)-np.eye(N))
check('T_inverse',T.conj().T-np.linalg.matrix_power(T,N-1))
nonhermitian=float(np.max(np.abs(T-T.conj().T)))
if nonhermitian<=.5: raise AssertionError('The N=8 translation must not be Hermitian')
check('H_hermiticity',H-H.conj().T)
check('H_T_commutator',H@T-T@H)
check('Fourier_orthonormality',F.conj().T@F-np.eye(N))
check('Fourier_completeness',F@F.conj().T-np.eye(N))
check('translation_eigen_residual',T@F-F*phases)
check('H_eigen_residual',H@F-F*energies)
check('H_diagonalization',F.conj().T@H@F-np.diag(energies))
check('direct_spectrum',np.linalg.eigvalsh(H)-np.sort(energies))
check('trace',np.trace(H)-N*E0)
check('sum_of_energies',sum(energies)-N*E0)
check('N8_observed_bandwidth',np.max(energies)-np.min(energies)-4*Delta)
check('BZ_label_periodicity',np.exp(-1j*(angles+2*np.pi))-phases)
check('real_hopping_plus_minus_degeneracy',energies-energies[(-indices)%N])
groups=[]
for energy in sorted(set(np.round(energies,12))):
    js=indices[np.isclose(energies,energy,atol=1e-11)].tolist()
    groups.append({'energy':float(energies[js[0]]),'j':js,'multiplicity':len(js)})
check('degeneracy_total',sum(g['multiplicity'] for g in groups)-N)

rows=[]
with (OUT/'4.3-N8平移相位与能级.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f)
    writer.writerow(['j','theta_raw','theta_first_BZ','k_for_a_1','translation_phase_real',
                     'translation_phase_imag','energy','energy_multiplicity'])
    for j in indices:
        mult=int(sum(np.isclose(energies,energies[j],atol=1e-12)))
        row=[int(j),float(angles[j]),float(folded[j]),float(folded[j]/a),
             float(phases[j].real),float(phases[j].imag),float(energies[j]),mult]
        rows.append(row);writer.writerow(row)

# Continuum Bloch-form illustration on a six-cell periodic ring.
cells=6;length=cells*a;k=np.pi/(3*a);G=2*np.pi/a
norm=np.sqrt(length*(1+.4**2+.2**2))
u=lambda x:(1+.4*np.exp(1j*G*x)+.2*np.exp(-2j*G*x))/norm
psi=lambda x:np.exp(1j*k*x)*u(x)
dxpsi=lambda x:1j*(k*np.exp(1j*k*x)+.4*(k+G)*np.exp(1j*(k+G)*x)
                       +.2*(k-2*G)*np.exp(1j*(k-2*G)*x))/norm
xs=np.linspace(0,length,6001)
ps=psi(xs);us=u(xs)
check('periodic_u',u(xs+a)-us)
check('Bloch_forward_relation',psi(xs+a)-np.exp(1j*k*a)*ps)
check('active_translation_relation',psi(xs-a)-np.exp(-1j*k*a)*ps)
check('periodic_density',np.abs(psi(xs+a))**2-np.abs(ps)**2)
check('six_cell_ring_boundary',psi(length)-psi(0))
check('Bloch_ring_normalization',np.trapezoid(np.abs(ps)**2,xs)-1)
weights=np.array([1,.4**2,.2**2])/(1+.4**2+.2**2)
wavevectors=np.array([k,k+G,k-2*G])
pmean=float(np.trapezoid(np.conj(ps)*(-1j*hbar*dxpsi(xs)),xs).real)
check('momentum_weights',sum(weights)-1)
check('mean_momentum_vs_Fourier',pmean-hbar*np.dot(weights,wavevectors))
check('mean_momentum_explicit',pmean-hbar*(k+G/15))
with (OUT/'4.3-布洛赫构造波形.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f);writer.writerow(['x','psi_real','psi_imag','u_real','u_imag','density'])
    for xx,pp,uu in zip(xs,ps,us):writer.writerow([xx,pp.real,pp.imag,uu.real,uu.imag,abs(pp)**2])

blue,red,purple,green='#2369a1','#b54237','#74469c','#2a7f64'
theta=np.linspace(-np.pi,np.pi,1201)
fig,ax=plt.subplots(figsize=(10.6,6.3),layout='constrained')
ax.plot(theta,-2*np.cos(theta),color=blue,lw=2,label=r'最近邻色散 $E=-2\cos\theta$')
ax.scatter(folded,energies,s=58,c=purple,zorder=3,label=r'$N=8$ 的8个允许态')
ax.scatter([np.pi],[2],s=68,facecolors='white',edgecolors=purple,lw=1.7,zorder=4)
for j in indices:
    shift=(0,11)
    if j==4:shift=(16,-22)
    if j==0:shift=(0,-22)
    ax.annotate(f'j={j}',(folded[j],energies[j]),xytext=shift,textcoords='offset points',ha='center',fontsize=10)
ax.annotate('与 j=4 相同的标签\n不另算第9个态',(np.pi,2),xytext=(1.3,1.9),
            arrowprops={'arrowstyle':'->','color':purple},ha='center',fontsize=10)
ax.set_title('有限周期环：8个独立态，5个不同能量',fontsize=16,fontweight='bold',pad=16)
ax.set_xlabel(r'$\theta=ka$，选取 $[-\pi,\pi)$');ax.set_ylabel('能量 E（Δ=1）')
ax.set_xlim(-np.pi-.14,np.pi+.14);ax.set_ylim(-2.55,2.8)
ax.set_xticks(np.arange(-4,5)*np.pi/4,[r'$-\pi$',r'$-3\pi/4$',r'$-\pi/2$',r'$-\pi/4$',r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$'])
ax.grid(alpha=.15);ax.legend(loc='upper center',frameon=False,ncols=2)
fig.savefig(OUT/'4.3-有限晶格离散谱与余弦带.png',dpi=165)
fig.savefig(OUT/'4.3-有限晶格离散谱与余弦带.svg');plt.close(fig)

ds=np.linspace(0,1.5,401)
fig,ax=plt.subplots(figsize=(10.6,6.3),layout='constrained')
ax.fill_between(ds,-2*ds,2*ds,color='#e8edf3',label='连续色散的能量范围；有限环仅取其中离散值')
colors=[blue,green,'#646d77',purple,red]
for group,color in zip(groups,colors):
    slope=group['energy'];ys=slope*ds
    ax.plot(ds,ys,color=color,lw=2.0)
    names=','.join(map(str,group['j']))
    ax.text(1.53,slope*1.5,f"j={names}；{group['multiplicity']}重",va='center',fontsize=10,color=color)
ax.scatter([0],[0],s=48,color='black',zorder=4)
ax.annotate('Δ=0：8重简并',(0,0),xytext=(.08,.7),fontsize=10,
            arrowprops={'arrowstyle':'->','color':'#444'})
ax.set_title('固定平移相位时，能量随耦合 Δ 线性变化',fontsize=16,fontweight='bold',pad=16)
ax.set_xlabel('最近邻耦合 Δ');ax.set_ylabel(r'能量 E（$E_0=0$）')
ax.set_xlim(-.04,1.98);ax.set_ylim(-3.45,3.45);ax.grid(alpha=.15)
ax.legend(loc='upper left',frameon=False,fontsize=10)
fig.savefig(OUT/'4.3-耦合增大与能级劈裂.png',dpi=165)
fig.savefig(OUT/'4.3-耦合增大与能级劈裂.svg');plt.close(fig)

fig,axs=plt.subplots(3,1,figsize=(10.8,8.7),sharex=True,layout='constrained')
fig.suptitle('构造的 Bloch 波形：周期因子、相位与概率密度',fontsize=16,fontweight='bold')
axs[0].plot(xs,ps.real,color=blue,lw=1.8,label=r'$\mathrm{Re}\,\psi_k$')
axs[0].plot(xs,ps.imag,color=red,lw=1.5,ls='--',label=r'$\mathrm{Im}\,\psi_k$')
axs[0].set_ylabel('波函数振幅');axs[0].legend(loc='upper right',frameon=False,ncols=2)
axs[0].text(.02,.92,r'$a=1,\quad k=\pi/3,\quad L=6a$',transform=axs[0].transAxes,
            va='top',fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
axs[1].plot(xs,us.real,color=green,lw=1.8,label=r'$\mathrm{Re}\,u_k$')
axs[1].plot(xs,us.imag,color=purple,lw=1.5,ls='--',label=r'$\mathrm{Im}\,u_k$')
axs[1].set_ylabel('周期因子');axs[1].legend(loc='upper right',frameon=False,ncols=2)
axs[2].plot(xs,abs(ps)**2,color='#424b57',lw=1.7,label=r'$|\psi_k|^2=|u_k|^2$')
axs[2].set_ylabel('概率密度');axs[2].set_xlabel('位置 x（长度单位 a）')
axs[2].legend(loc='upper right',frameon=False)
for ax in axs:
    ax.grid(alpha=.13);ax.set_xlim(0,6)
    ax.axvspan(1,2,color='#e8edf3',alpha=.7,zorder=-1)
    ax.axvspan(3,4,color='#e8edf3',alpha=.7,zorder=-1)
    ax.axvspan(5,6,color='#e8edf3',alpha=.7,zorder=-1)
axs[0].set_ylim(-.7,.75);axs[1].set_ylim(-.3,.8);axs[2].set_ylim(0,.43)
axs[2].text(.02,.06,'只示范Bloch变换；未宣称是某个指定势的能量本征解',transform=axs[2].transAxes,
            fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
fig.savefig(OUT/'4.3-布洛赫复波函数与周期因子.png',dpi=165)
fig.savefig(OUT/'4.3-布洛赫复波函数与周期因子.svg');plt.close(fig)

result={'status':'PASS','finite_ring':{'N':N,'E0':E0,'Delta':Delta,'a':a,'hbar':hbar,
        'T_minus_Tdag_max_abs':nonhermitian,'trace':float(np.trace(H).real),
        'bandwidth':float(np.ptp(energies)),'energy_groups':groups,
        'states_csv':'4.3-N8平移相位与能级.csv'},
        'Bloch_construction':{'a':a,'k':k,'G':G,'ring_length':length,
          'u_formula':'(1+0.4 exp(i G x)+0.2 exp(-2i G x))/sqrt(1.2 L)',
          'momentum_over_hbar':wavevectors.tolist(),'momentum_weights':weights.tolist(),
          'mean_momentum_over_hbar':pmean/hbar,
          'warning':'A constructed Bloch-form wavefunction, not a solved potential eigenfunction.'},
        'checks':checks}
(OUT/'4.3-有限晶格与布洛赫核验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','groups':len(checks),'max_residual':max(v['max_abs_error'] for v in checks.values())}))
