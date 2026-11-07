"""教学模型：两次自旋脉冲；有限时间核与可精确积分的洛伦兹谱权。"""
from pathlib import Path
import json,csv,os,sys
out=Path(__file__).resolve().parent
root=out.parent.parent
if sys.version_info[:2]==(3,12) and (root/'tmp/section3_1_deps').exists():
    sys.path.insert(0,str(root/'tmp/section3_1_deps'))
os.environ['MPLCONFIGDIR']=str(root/'tmp/section5_7_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
cn=Path('C:/Windows/Fonts/msyh.ttc')
if cn.exists(): font_manager.fontManager.addfont(str(cn))
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':12,
                     'axes.unicode_minus':False,'svg.fonttype':'none',
                     'axes.spines.right':False,'axes.spines.top':False})
checks=[]
def ck(name,diff,tol):
    err=float(np.max(np.abs(diff)))
    checks.append(dict(name=name,absolute_error=err,tolerance=tol,passed=err<tol))
def K(y,u):
    return u*u*np.sinc(y*u/(2*np.pi))**2
def p(u):
    return 2*np.pi*.002*(u+np.expm1(-u))
def table(name,head,rows):
    with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(head);w.writerows(rows)
# Observe sigma_x following an x-then-y vs y-then-x sequence.
I=np.eye(2,dtype=complex);sx=np.array([[0,1],[1,0]],complex)
sy=np.array([[0,-1j],[1j,0]],complex);sz=np.diag([1,-1]);a=b=np.pi/4
R=lambda S,q: I*np.cos(q)-1j*S*np.sin(q)
plus=np.array([1,1])/np.sqrt(2);up=np.array([1,0])
for tag,U,p0 in [('先x后y',R(sy,b)@R(sx,a),.5),('先y后x',R(sx,a)@R(sy,b),1)]:
    ck(tag+' 的 x+ 概率',abs(plus.conj()@U@up)**2-p0,1e-12)
    ck(tag+' 的精确幺正性',U.conj().T@U-I,1e-12)
# Same-axis chronological integrals give 1/2, cross terms encode ordering.
for e in (.04,.02,.01):
    exact=R(sy,e)@R(sx,e)
    app=I-1j*e*(sx+sy)-e*e*(I+sy@sx)
    ck(f'按时间排序的二阶近似，脉冲面积 {e}',exact-app,2*e**3)
    wrong=I-1j*e*(sx+sy)-e*e*(I+(sx@sy+sy@sx)/2)
    ck('普通指数漏掉的对易子',app-wrong+e*e*(sy@sx-sx@sy)/2,1e-14)
y=np.linspace(-200,200,80001)
for u in (.1,.5,1,2,4,8):
    direct=np.trapezoid(.002*K(y,u)/(1+y*y),y)
    ck(f'直接谱积分与闭式，Ωt={u}',direct-p(u),2e-9)
    ck('首零点',K(2*np.pi/u,u),1e-25)
    ck('峰高',K(0,u)-u*u,1e-13)
    h=2e-5
    ck('瞬时斜率', (p(u+h)-p(u-h))/(2*h)-2*np.pi*.002*(1-np.exp(-u)),5e-12)
u=np.linspace(0,8,401); x=np.linspace(-12,12,1201)
table('5.7-共振核.csv',['omega/Omega','Omega² K, Omega t=1','Omega² K, Omega t=2','Omega² K, Omega t=4'],zip(x,*[K(x,s) for s in (1,2,4)]))
table('5.7-连续谱总概率.csv',['Omega*t','P leading','w/Omega instantaneous','golden rate/Omega'],zip(u,p(u),2*np.pi*.002*(1-np.exp(-u)),np.full_like(u,2*np.pi*.002)))
fig,(ax,ay)=plt.subplots(1,2,figsize=(11.2,4.45),layout='constrained')
for s,c in [(1,'#b56324'),(2,'#3978a2'),(4,'#00675e')]:
    ax.plot(x,K(x,s),c=c,lw=2,label=rf'$\Omega t={s}$')
ax.set(xlabel=r'失谐 $\omega/\Omega$',ylabel=r'谱核 $\Omega^2 K_t$',title='固定时间，看不同末态');ax.legend(frameon=False)
ax.annotate('首个零点',xy=(np.pi/2,0),xytext=(3.6,5.7),arrowprops=dict(arrowstyle='->',color='#424242'),fontsize=10)
ay.plot(u,p(u),c='#00675e',lw=2.7,label='谱积分：总概率')
ay.plot(u,2*np.pi*.002*u,c='#9a5831',ls='--',lw=1.7,label='黄金法则斜率对应的直线')
ay.set(xlabel=r'经过的时间 $\Omega t$',ylabel='总跃迁概率（无量纲）',title='固定谱权，积分所有末态',ylim=(0,.105));ay.legend(frameon=False,loc='upper left')
for axx in (ax,ay): axx.grid(axis='y',alpha=.16)
for ext in ('svg','png'): fig.savefig(out/f'5.7-共振峰与连续谱总概率.{ext}',dpi=175)
plt.close(fig)
(out/'5.7-时间排序与黄金法则核验.json').write_text(json.dumps({'checks':checks,'spectral_weight':'Lorentzian at J0/Omega=0.002; effective smooth continuum model; no experimental data','P_u8':p(8.),'dP_u8_div_asymptotic':1-np.exp(-8)},ensure_ascii=False,indent=2),encoding='utf-8')
assert all(c['passed'] for c in checks),[c for c in checks if not c['passed']]
print(len(checks),'checks passed')

