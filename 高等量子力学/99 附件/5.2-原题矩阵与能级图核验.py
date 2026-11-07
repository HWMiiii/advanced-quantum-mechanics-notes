"""§5.2原题5.6/7/9/12/13/15的矩阵检查及5.13明确要求的能级图。"""
from pathlib import Path
import sys, os, json, math, csv

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
if sys.version_info[:2] == (3,12):
    sys.path.insert(0, str(ROOT/'tmp/section3_1_deps'))
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/section5_2_exercises/mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'svg.fonttype':'none'})

checks=[]
def check(name,actual,expected,tol=1e-10):
    err=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    checks.append(dict(name=name,residual=err,tolerance=tol,passed=err<tol))

gamma=256/(81*np.pi**4)
X12=-16/(9*np.pi**2)
check('5.6/5.13 非对角元等于X12平方',X12**2,gamma)
Q=np.array([[1,1],[1,-1]])/np.sqrt(2)
W=np.array([[.25,gamma],[gamma,.25]])
check('5.6/5.13 对称反对称组合对角化',Q.T@W@Q,np.diag([.25+gamma,.25-gamma]))

def well(N,g):
    n=np.arange(1,N+1,dtype=float)
    X=np.zeros((N,N))
    for i,m in enumerate(n):
        for j,k in enumerate(n):
            if i==j:X[i,j]=.5
            elif int(m+k)%2:X[i,j]=-8*m*k/(np.pi**2*(m*m-k*k)**2)
    H=np.diag((n[:,None]**2+n[None,:]**2).ravel())+g*np.kron(X,X)
    return np.linalg.eigvalsh(H)[:3]

well_rows=[]
for g in [.1,.2]:
    a=well(12,g); b=well(16,g)
    first=np.array([2+g/4,5+g*(.25-gamma),5+g*(.25+gamma)])
    check(f'5.13 有限基12与16收敛 g={g}',a,b,2e-9)
    check(f'5.13 一阶能量余项有界 g={g}',b,first,.02*g*g)
    for label,e,p in zip(['基态11','反对称12-21','对称12+21'],b,first):
        well_rows.append(dict(g=g,state=label,finite_basis_energy=float(e),first_order=float(p),residual=float(e-p)))

delta=.04
exact=np.array([(.5*np.sqrt(1+delta)+.5*np.sqrt(1-delta)),
                1.5*np.sqrt(1+delta)+.5*np.sqrt(1-delta),
                .5*np.sqrt(1+delta)+1.5*np.sqrt(1-delta)])
first=np.array([1,2+delta/2,2-delta/2])
check('5.7 正常坐标精确解的一阶余项',exact,first,delta**2)

eps=.03; Omega=.02
Lx=np.array([[0,0,0],[0,0,-1j],[0,1j,0]])
M=np.diag([eps/2,eps/2,3*eps/2])-Omega*Lx
pred=np.sort([eps/2,eps+np.sqrt((eps/2)**2+Omega**2),eps-np.sqrt((eps/2)**2+Omega**2)])
check('5.9 三维振子一阶矩阵谱',np.linalg.eigvalsh(M),pred)
check('5.9 厄米矩阵',M,M.conj().T)

# Real p-orbital angular moments use <nx^4>=1/5 and <nx^2 ny^2>=1/15.
check('5.12 实p轨道角因子',3*(1/5-1/15),2/5)

E1=0.;E2=1.;a=.03+.02j;b=-.04+.01j
rho=np.sqrt(abs(a)**2+abs(b)**2)
H=np.array([[E1,0,a],[0,E1,b],[a.conjugate(),b.conjugate(),E2]])
B=np.array([a,b,0])/rho
D=np.array([b.conjugate(),-a.conjugate(),0])/rho
three=np.array([0.,0.,1.])
check('5.15 暗态精确不移动',H@D,E1*D)
check('5.15 亮暗态正交归一',np.column_stack([D,B,three]).conj().T@np.column_stack([D,B,three]),np.eye(3))
Epm=np.array([(E1+E2)/2-np.sqrt((E2-E1)**2/4+rho**2),E1,(E1+E2)/2+np.sqrt((E2-E1)**2/4+rho**2)])
check('5.15 三态精确能谱',np.linalg.eigvalsh(H),Epm)
W2=-np.array([[abs(a)**2,a*b.conjugate()],[b*a.conjugate(),abs(b)**2]])/(E2-E1)
check('5.15 二阶有效矩阵的非零与零根',np.linalg.eigvalsh(W2),[-rho**2/(E2-E1),0])

# The original problem 5.13(b)(iii) explicitly requires a level diagram.
g=.2
fig,(ax,zoom)=plt.subplots(1,2,figsize=(12.8,6.2),gridspec_kw={'width_ratios':[1.03,1]})
color_ground='#276C91'; color_plus='#C5652E';color_minus='#436B4C'
for axis in [ax,zoom]:
    axis.set_xlim(-.25,1.4);axis.set_xticks([0,1],['无微扰','加入微扰'])
    axis.grid(axis='y',alpha=.16);axis.spines[['top','right']].set_visible(False)

for initial,final,col in [(2,2+g/4,color_ground),(5,5+g*(.25+gamma),color_plus),(5,5+g*(.25-gamma),color_minus)]:
    ax.plot([-.13,.13],[initial,initial],color=col,lw=2)
    ax.plot([.87,1.13],[final,final],color=col,lw=2)
    ax.plot([.13,.87],[initial,final],color=col,lw=1.4)
ax.set_ylim(1.75,5.55);ax.set_ylabel(r'能量 $E/E_*$，$E_*=\pi^2\hbar^2/(2ma^2)$')
ax.set_title('三个最低态的完整能级位置',pad=12)
ax.text(-.18,1.91,r'$|11\rangle$',color=color_ground,va='top')
ax.text(-.18,5.13,'二重简并：可选 |+〉、|-〉',fontsize=10)
ax.text(1.16,2+g/4,'基态',color=color_ground,va='center',fontsize=10)
ax.text(.96,5.27,'第一激发双重态\n分裂放大见右图',ha='center',fontsize=10)

for s,coef,col,lab in [(1,.25+gamma,color_plus,r'$|+\rangle=(|12\rangle+|21\rangle)/\sqrt{2}$'),(-1,.25-gamma,color_minus,r'$|-\rangle=(|12\rangle-|21\rangle)/\sqrt{2}$')]:
    zoom.plot([-.13,.13],[0,0],color=col,lw=2)
    zoom.plot([.87,1.13],[coef,coef],color=col,lw=2.3)
    zoom.plot([.13,.87],[0,coef],color=col,lw=1.5)
    zoom.text(-.18,.335 if s==1 else .305,lab,color=col,ha='left',fontsize=10)
    zoom.text(1.16,coef,f'{coef:.5f}',color=col,va='center',fontsize=9)
zoom.set_ylim(-.04,.36)
zoom.set_ylabel(r'激发态能移 $[E-5E_*]/(\lambda a^2)$')
zoom.set_title('双重态分裂放大与正确连接',pad=12)
zoom.axhline(.25,color='#999999',linestyle=':',linewidth=1)
fig.suptitle('教材5.13(b)(iii)：二维方阱的简并微扰能级图',fontsize=17,y=.97)
fig.text(.5,.035,r'示意取 $g=\lambda a^2/E_*=0.2>0$；无扰简并处选取 |±〉基，不能把单独 |12〉、|21〉各自唯一接到一条能支。',ha='center',fontsize=10)
fig.subplots_adjust(left=.075,right=.96,bottom=.16,top=.84,wspace=.34)
fig.savefig(OUT/'5.2-习题5.13-二维方阱能级连接.png',dpi=180)
fig.savefig(OUT/'5.2-习题5.13-二维方阱能级连接.svg')
plt.close(fig)
with (OUT/'5.2-原题方阱数值核验.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=well_rows[0].keys());w.writeheader();w.writerows(well_rows)
res=dict(description='原题5.13要求的图已生成；数值参数为透明声明的教学核验',checks=checks,well=well_rows,gamma=float(gamma),all_passed=all(x['passed'] for x in checks))
(OUT/'5.2-原题矩阵与能级图核验.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
if not res['all_passed']:raise SystemExit('verification failed')
