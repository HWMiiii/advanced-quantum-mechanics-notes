"""原题5.18、5.24—5.26：量级计算、独立矩阵核验、求积与八态能级图。
原题来源：PDF378—380／印刷356—358；数据不是实验测量。
常数取NIST CODATA 2022。5.24 A,B均有能量量纲；图中x=B/A。
运行依赖 numpy、matplotlib；输出保存在本脚本所在目录。
"""
from pathlib import Path
from math import pi,sqrt,factorial
import os,sys,json,csv
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
if sys.version_info[:2]==(3,12):sys.path.insert(0,str(ROOT/'tmp/section3_1_deps'))
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/section5_3_exercises/mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,
                     'axes.unicode_minus':False,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
checks=[]
def check(name,res,tol=1e-10):
    value=float(np.max(np.abs(res)))
    checks.append({'name':name,'residual':value,'tolerance':tol,'passed':bool(value<tol)})
def csvout(name,header,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.writer(fp);writer.writerow(header);writer.writerows(rows)
def figout(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

# 5.18. Leading infinite-nuclear-mass textbook model, not precision hydrogen data.
C={'source':'https://physics.nist.gov/cuu/pdf/all.pdf','adjustment':'CODATA 2022',
   'alpha':7.2973525643e-3,'a0_m':5.29177210544e-11,'electron_mass_kg':9.1093837139e-31,
   'c_m_s':299792458.,'h_J_s':6.62607015e-34,'elementary_charge_C':1.602176634e-19,
   'muB_J_T':9.2740100657e-24,'NA_mol_inv':6.02214076e23,'mu0_SI':1.25663706127e-6}
alpha,a0,me,c,h,qe,muB,NA,mu0=[C[k] for k in ['alpha','a0_m','electron_mass_kg','c_m_s','h_J_s','elementary_charge_C','muB_J_T','NA_mol_inv','mu0_SI']]
mc2=me*c*c;scale=mc2*alpha**4
lam=656e-9
gap3=scale/108;gap2=scale/32
dlam=lam**2*gap3/(h*c)
# Symmetric photon energies give a second, exact-in-conversion check.
photon=h*c/lam
dlam_exact=h*c/(photon-gap3/2)-h*c/(photon+gap3/2)
check('5.18(a) 线性与精确波长换算相对差',dlam/dlam_exact-1,1e-9)
ks=-13*scale/128;kp=-7*scale/384;kgap=kp-ks
check('5.18(b) 两个动能修正之差=mc2 alpha4/12',(kgap/scale)-1/12)
Efull=kgap/(6*qe*a0);Eoneside=kgap/(3*qe*a0)
p518={'wavelength_nm':lam*1e9,'3p_doublet_gap_eV':gap3/qe,'3p_doublet_gap_GHz':gap3/h/1e9,
      'wavelength_separation_nm':dlam*1e9,'exact_conversion_nm':dlam_exact*1e9,
      'resolving_power':lam/dlam,'2s_kinetic_shift_eV':ks/qe,'2p_kinetic_shift_eV':kp/qe,
      'kinetic_gap_eV':kgap/qe,'electric_full_split_V_m':Efull,'electric_full_split_kV_cm':Efull/1e5,
      'electric_one_side_kV_cm':Eoneside/1e5,'voltage_at_1mm_full_V':Efull*.001,
      'voltage_at_1mm_one_side_V':Eoneside*.001,
      '2p_spin_orbit_gap_eV':gap2/qe,'2p_crossover_T':gap2/muB,'3p_crossover_T':gap3/muB,
      'weak_example_T':.001,'weak_ratio_to_2p_SO':muB*.001/gap2,
      'strong_example_T':10.,'strong_ratio_to_2p_SO':muB*10/gap2,
      'weak_example_solenoid_A_turns_per_m':.001/mu0,
      'strong_10T_Zeeman_eV':10*muB/qe}
csvout('5.3-5.18量级计算.csv',['quantity','value'],p518.items())

# 5.24. Build L.S and magnetic operator independently in the product basis.
# Product basis: s0 up, s0 down, p1 up, p1 down, p0 up, p0 down, p-1 up, p-1 down.
prod=[(0,0,.5),(0,0,-.5),(1,1,.5),(1,1,-.5),(1,0,.5),(1,0,-.5),(1,-1,.5),(1,-1,-.5)]
LS=np.diag([ml*ms for l,ml,ms in prod]);Z=np.diag([ml+2*ms for l,ml,ms in prod])
for i,(l,ml,ms) in enumerate(prod):
    for j,(lp,mlp,msp) in enumerate(prod):
        if l==lp and mlp==ml+1 and msp==ms-1:
            LS[i,j]=LS[j,i]=sqrt(l*(l+1)-ml*(ml+1))/2
U=np.zeros((8,8));U[0,0]=1;U[1,1]=1;U[2,2]=1;U[7,3]=1
U[4,4]=sqrt(2/3);U[3,4]=sqrt(1/3);U[4,5]=-sqrt(1/3);U[3,5]=sqrt(2/3)
U[6,6]=sqrt(1/3);U[5,6]=sqrt(2/3);U[6,7]=-sqrt(2/3);U[5,7]=sqrt(1/3)
check('5.24 CG变换酉性',U.T@U-np.eye(8))
def coupled(A,B):
    M=np.diag([B,-B,A/2+2*B,A/2-2*B,0.,0.,0.,0.])
    for first,m in [(4,.5),(6,-.5)]:
        M[first:first+2,first:first+2]=[[A/2+4*m*B/3,-sqrt(2)*B/3],[-sqrt(2)*B/3,-A+2*m*B/3]]
    return M
def eigbranches(x):
    x=np.asarray(x)
    out=[x,-x,.5+2*x,.5-2*x]
    for m in [.5,-.5]:
        q=.25*np.sqrt(9+8*m*x+4*x*x)
        out.extend([-.25+m*x+q,-.25+m*x-q])
    return np.array(out)
for A,B in [(1,0),(1,.001),(1,.3),(1,1),(1,2),(1,10),(1,100),(.31,-2.7)]:
    check(f'5.24 独立乘积基矩阵转耦合基 A={A},B={B}',U.T@(A*LS+B*Z)@U-coupled(A,B))
    check(f'5.24 解析八能量对数值对角化 A={A},B={B}',np.sort(A*eigbranches(B/A))-np.linalg.eigvalsh(A*LS+B*Z))
    check(f'5.24 迹为零 A={A},B={B}',np.trace(coupled(A,B)))
small=1e-5
deriv=(eigbranches(small)-eigbranches(-small))/(2*small)
weak_slopes=np.array([1,-1,2,-2,2/3,1/3,-2/3,-1/3])
check('5.24 弱场Landé斜率',deriv-weak_slopes,1e-8)
large=1e5
strong_slopes=np.array([1,-1,2,-2,1,0,0,-1])
strong_offsets=np.array([0,0,.5,.5,0,-.5,-.5,0])
strong_err=eigbranches(large)-large*strong_slopes-strong_offsets
check('5.24 强场乘积态截距（B/A=1e5）',strong_err,6e-6)
x=np.linspace(0,12,1201);y=eigbranches(x)
branchids=['2s_up','2s_down','2p_stretched_up','2p_stretched_down','mplus_upper','mplus_lower','mminus_upper','mminus_lower']
csvout('5.3-5.24八态能移.csv',['B_over_A']+[s+'_Delta_over_A' for s in branchids],zip(x,*y))
fig,axs=plt.subplots(1,2,figsize=(13.2,5.5),layout='constrained',gridspec_kw={'width_ratios':[1.18,1]})
labels=[r'2s: $m=+1/2$',r'2s: $m=-1/2$',r'2p: $m=+3/2$',r'2p: $m=-3/2$',
        r'2p: $m=+1/2$, 上支',r'2p: $m=+1/2$, 下支',r'2p: $m=-1/2$, 上支',r'2p: $m=-1/2$, 下支']
colors=['#b73747','#295e9e','#cb692d','#21837a','#855096','#ba8d24','#ca659d','#5c6f37']
styles=['--','--','-','-','-','-','-','-']
for ax in axs:
    for i in range(8):ax.plot(x,y[i],color=colors[i],ls=styles[i],lw=1.9,label=labels[i],zorder=4 if i<2 else 2)
    ax.axhline(0,color='#69717a',lw=.6,alpha=.4);ax.grid(alpha=.18)
    ax.set(xlabel=r'$B/A$（两者均为能量）',ylabel=r'$\Delta/A$')
axs[0].set(xlim=(0,2.2),ylim=(-4.2,5.3),title='弱场到中间区：8 条能级分支')
axs[1].set(xlim=(0,12),ylim=(-24.5,25),title='强场区：按 5 组斜率分开，仍有 8 态')
axs[0].legend(fontsize=9,ncol=2,loc='upper left',framealpha=.92)
for slope,count in [(-2,1),(-1,2),(0,2),(1,2),(2,1)]:
    axs[1].text(12.2,12*slope+.1,f'斜率 {slope:+d}：{count} 态',ha='left',va='center',fontsize=9)
fig.suptitle('原题 5.24：自旋—轨道耦合与线性塞曼项的联合对角化（A > 0）',fontsize=13)
figout(fig,'5.3-5.24八态塞曼跨越')

# 5.25/26. Independent Gauss-Laguerre radial quadrature x=2 Z r/a0.
nodes,weights=np.polynomial.laguerre.laggauss(80)
def hydrogenic_moments(Zeff):
    norm=np.sum(weights*nodes**2/2)
    r2=np.sum(weights*nodes**4/(8*Zeff**2))
    return float(norm),float(r2)
normH,r2H=hydrogenic_moments(1.)
zeff=27/16;normHe,r2He=hydrogenic_moments(zeff)
check('5.25 H1s径向概率归一化',normH-1)
check('5.25 H1s <r2>/a0^2=3',r2H-3)
check('5.26 给定单电子径向概率归一化',normHe-1)
check('5.26 给定态 <r2>/a0^2=3/Zeff^2',r2He-3/zeff**2)
a0cm=a0*100
chiH=-alpha**2*a0cm**3/2
chiHe=alpha**2*a0cm**3/zeff**2
chiHmol=chiH*NA;chiHemol=chiHe*NA
measured=1.88e-6
p525={'r2_over_a0_squared':r2H,'chi_atom_cgs_cm3':chiH,'chi_molar_cgs_cm3_mol':chiHmol,
      'physical_chi_molar_SI_m3_mol':4*pi*1e-6*chiHmol}
p526={'Zeff':zeff,'one_electron_r2_over_a0_squared':r2He,'chi_energy_coefficient_atom_cgs_cm3':chiHe,
      'chi_energy_coefficient_molar_cgs_cm3_mol':chiHemol,'physical_chi_molar_cgs_cm3_mol':-chiHemol,
      'physical_chi_molar_SI_m3_mol':-4*pi*1e-6*chiHemol,
      'textbook_measured_magnitude_cgs_cm3_mol':measured,'predicted_to_measured':chiHemol/measured,
      'relative_error_percent':100*(chiHemol/measured-1)}
csvout('5.3-5.25与5.26抗磁计算.csv',['exercise','quantity','value'],
       [(no,k,v) for no,vals in [('5.25',p525),('5.26',p526)] for k,v in vals.items()])
summary={'sources':{'5.18':'PDF378—379／印刷356—357','5.24—5.26':'PDF380／印刷358'},
         'constants':C,'5.18':p518,'5.24':{'basis_order':['s+','s-','p3/2+3/2','p3/2-3/2','p3/2+1/2','p1/2+1/2','p3/2-1/2','p1/2-1/2'],
            'LS_in_product_basis':LS.tolist(),'magnetic_in_product_basis':Z.tolist(),'coupled_transformation_U':U.tolist(),
            'weak_slopes':weak_slopes.tolist(),'strong_slopes':strong_slopes.tolist(),'strong_offsets':strong_offsets.tolist(),
            'strong_offset_check_x':large,'strong_offset_errors':strong_err.tolist(),'plot_points':len(x)},
         '5.25':p525,'5.26':p526,'checks':checks,'check_count':len(checks),
         'passed':all(p['passed'] for p in checks),'max_residual':max(p['residual'] for p in checks)}
(OUT/'5.3-5.18与5.24至5.26核验结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['5.18','5.25','5.26','check_count','passed','max_residual']},ensure_ascii=False,indent=2))
if not summary['passed']:raise SystemExit('Verification failed')
