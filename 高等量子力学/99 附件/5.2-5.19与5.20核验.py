"""原题5.19、5.20的解析结果交叉核验与教学图。
来源：教材PDF379／印刷357；正文PDF321／印刷299。
所有图形参数为教学选取，不计为新增原题。
"""
from pathlib import Path
from fractions import Fraction as Q
from math import factorial, sqrt, pi
import sys,os,json,csv
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
if sys.version_info[:2]==(3,12):sys.path.insert(0,str(ROOT/'tmp/section3_1_deps'))
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/section5_2_exercises/mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10.5,
                     'axes.unicode_minus':False,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
checks=[]
def check(name,res,tol=1e-9):
    val=float(np.max(np.abs(res)))
    checks.append({'name':name,'residual':val,'tolerance':tol,'passed':bool(val<tol)})
def savefig(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)
def csvout(name,header,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as fp:
        wr=csv.writer(fp);wr.writerow(header);wr.writerows(rows)

# 5.19: numerical integration of <2s|z|2p,m=0>.
x,w=np.polynomial.laguerre.laggauss(80)
radial_sp=float(np.sum(w*x**3*(2-x)*x/(8*np.sqrt(3))))
z_sp=radial_sp/np.sqrt(3)
check('5.19 2s—2p0 z矩阵元=-3 a0',z_sp+3)
check('5.19 自旋角函数系数给sqrt3',-z_sp/np.sqrt(3)-np.sqrt(3))
u=np.array([[0,0,-1,0],[0,0,0,1],[1,0,0,0],[0,-1,0,0]],float)
check('5.19 时间反演平方=-I',u@u+np.eye(4))
lamb_samples=[]
for g in [.01,.1,1,10]:
    h=np.zeros((4,4));h[:2,:2]=[[1,g],[g,0]];h[2:,2:]=[[1,-g],[-g,0]]
    en,vec=np.linalg.eigh(h)
    low=.5-np.sqrt(.25+g*g);high=.5+np.sqrt(.25+g*g)
    check(f'5.19 g/Delta={g} 双重精确谱',en-[low,low,high,high])
    check(f'5.19 g/Delta={g} 时间反演对称',u@h.conj()@u.T-h)
    theta=.5*np.arctan2(2*g,1)
    up=np.array([np.cos(theta),np.sin(theta),0,0])
    down=u@up.conj()
    check(f'5.19 g/Delta={g} Kramers伙伴同能',h@down-high*down)
    check(f'5.19 g/Delta={g} Kramers伙伴正交',np.vdot(up,down))
    lamb_samples.append({'g_over_Delta':g,'E_low_over_Delta':low,'E_high_over_Delta':high,
                         's_branch_shift':high-1,'p_branch_shift':low,'weak_shift_magnitude':g*g})
gg=np.linspace(0,4,401)
lo=.5-np.sqrt(.25+gg**2);hi=.5+np.sqrt(.25+gg**2)
csvout('5.2-5.19兰姆跨越扫描.csv',['g_over_Delta','E_low_over_Delta','E_high_over_Delta','s_shift','p_shift'],zip(gg,lo,hi,hi-1,lo))
fig,axs=plt.subplots(1,2,figsize=(12.5,4.6),layout='constrained')
axs[0].plot(gg,hi,lw=2.2,color='#bd5729',label='上支（弱场接到 2s₁/₂）')
axs[0].plot(gg,lo,lw=2.2,color='#176b7a',label='下支（弱场接到 2p₁/₂）')
axs[0].plot(gg,.5+gg,'--',color='#bd5729',alpha=.65,label='强混合极限')
axs[0].plot(gg,.5-gg,'--',color='#176b7a',alpha=.65)
axs[0].set(xlabel='耦合 g / 兰姆能量差 ΔL',ylabel='能量 / ΔL（2p 零点）',title='每条线仍有两个 Kramers 伙伴')
small=np.linspace(0,.3,121)
shift=np.sqrt(.25+small**2)-.5
axs[1].plot(small,shift,color='#176b7a',lw=2.2,label='精确上移量')
axs[1].plot(small,small**2,'--',color='#bd5729',label='弱场：g² / ΔL²')
axs[1].set(xlabel='耦合 g / ΔL',ylabel='上支能移 / ΔL',title='弱场区首先呈二次响应')
for ax in axs:ax.legend(fontsize=9);ax.grid(alpha=.2)
fig.suptitle('原题 5.19：近简并两组能级的精确对角化；仍须与 2p₃/₂ 保持足够能隙',fontsize=12.5)
savefig(fig,'5.2-5.19兰姆位移与斯塔克跨越')

# 5.20: independent angular and radial quadratures.
ug,wg=np.polynomial.legendre.leggauss(80)
def angular(l,m):
    c=np.zeros(l+1);c[l]=1
    dl=np.polynomial.legendre.legder(c,m)
    p=(-1)**m*(1-ug**2)**(m/2)*np.polynomial.legendre.legval(ug,dl)
    return sqrt((2*l+1)/(4*pi)*factorial(l-m)/factorial(l+m))*p
def cos_element(l,lp,m):return 2*pi*np.sum(wg*angular(l,m)*ug*angular(lp,m))
for l,lp,m,exact in [(0,1,0,1/sqrt(3)),(1,2,0,2/sqrt(15)),(1,2,1,1/sqrt(5)),(2,3,0,3/sqrt(35)),(2,3,2,1/sqrt(7))]:
    check(f'5.20 角积分 l={l},{lp};m={m}',cos_element(l,lp,m)-exact)
r=1.5*x
rs=2/(81*np.sqrt(3))*(2*r*r-18*r+27)
rp=4/(81*np.sqrt(6))*r*(6-r)
rd=4/(81*np.sqrt(30))*r*r
measure=w*(1.5*r*r)
for l,rr in enumerate([rs,rp,rd]):check(f'5.20 R3{l} 归一化',np.sum(measure*rr*rr)-1)
z01=float(np.sum(measure*r*rs*rp))*cos_element(0,1,0)
z12=float(np.sum(measure*r*rp*rd))*cos_element(1,2,0)
z12m1=float(np.sum(measure*r*rp*rd))*cos_element(1,2,1)
check('5.20 <3s|z|3p0>=-3sqrt6',z01+3*np.sqrt(6))
check('5.20 <3p0|z|3d0>=-3sqrt3',z12+3*np.sqrt(3))
check('5.20 <3p1|z|3d1>=-9/2',z12m1+4.5)
zblock=np.array([[0,-3*np.sqrt(6),0],[-3*np.sqrt(6),0,-3*np.sqrt(3)],[0,-3*np.sqrt(3),0]])
states=np.column_stack([[1/np.sqrt(3),1/np.sqrt(2),1/np.sqrt(6)],
                        [1/np.sqrt(3),0,-np.sqrt(2/3)],
                        [1/np.sqrt(3),-1/np.sqrt(2),1/np.sqrt(6)]])
check('5.20 m0 三个态正交归一',states.T@states-np.eye(3))
check('5.20 m0 三个本征方程',zblock@states-states@np.diag([-9,0,9]))
check('5.20 m1 二阶矩阵谱',np.linalg.eigvalsh([[0,-4.5],[-4.5,0]])-[-4.5,4.5])

# Exact rational polynomial verification of zero branches' inhomogeneous equations.
def mul(p,q):
    out={}
    for k,c in p.items():
        for l,d in q.items():out[k+l]=out.get(k+l,Q(0))+c*d
    return out
def add(p,q):
    out=p.copy()
    for k,c in q.items():out[k]=out.get(k,Q(0))+c
    return {k:c for k,c in out.items() if c}
def scale(p,c):return {k:v*c for k,v in p.items()}
def op(p,l):
    out={}
    for k,c in p.items():
        out[k-2]=out.get(k-2,Q(0))+c*Q(k*(k+1)-l*(l+1),2)
        out[k-1]=out.get(k-1,Q(0))+c*Q(2-k,3)
    return {k:c for k,c in out.items() if c}
def integ(p):return sum(c*Q(factorial(k))*Q(3,2)**(k+1) for k,c in p.items())
def residual_dict(p):return float(max([abs(v) for v in p.values()]+[Q(0)]))
q={2:Q(1),1:Q(-6)};pp=scale(q,Q(-1));base={4:Q(-3),3:Q(9)};r2poly={2:Q(1)}
t=-integ(mul(r2poly,mul(pp,base)))/integ(mul(r2poly,mul(pp,q)))
p1=add(base,scale(q,t));s1={3:Q(2),2:Q(-30),1:Q(45)}
p3={4:Q(-3,2),3:Q(-18)};s3={3:Q(1)}
check('5.20 零m0 l1非齐次方程（有理精确）',residual_dict(add(op(p1,1),scale(s1,Q(-1)))))
check('5.20 零分支l3非齐次方程（有理精确）',residual_dict(add(op(p3,3),scale(s3,Q(-1)))))
check('5.20 l1一阶修正与3p正交（有理精确）',float(integ(mul(r2poly,mul(pp,p1)))))
check('5.20 零m0源与3p正交（有理精确）',float(integ(mul(r2poly,mul(pp,s1)))))
i1=Q(4,405**2*3)*integ(mul(r2poly,mul(s1,p1)))
i3=Q(16,405**2*7)*integ(mul(r2poly,mul(s3,p3)))
im2=Q(16,81**2*210)*integ(mul(r2poly,mul(s3,p3)))
check('5.20 零m0二阶能量系数=-3483/4',float(i1+i3+Q(3483,4)))
check('5.20 零m2二阶能量系数=-1377/2',float(im2+Q(1377,2)))
c1=2/(405*np.sqrt(3));c3=-4/(405*np.sqrt(7));cm2=4/(81*np.sqrt(210))
s1val=c1*r*(2*r*r-30*r+45)
p1val=c1*(-3*r**4+9*r**3+float(t)*(r*r-6*r))
s3val=c3*r**3;p3val=-1.5*c3*r**3*(r+12)
im0_num=float(np.sum(measure*(s1val*p1val+s3val*p3val)))
im2_num=float(np.sum(measure*(cm2*r**3)*(-1.5*cm2*r**3*(r+12))))
check('5.20 零m0独立Gauss-Laguerre积分',im0_num-float(i1+i3),1e-7)
check('5.20 零m2独立Gauss-Laguerre积分',im2_num-float(im2),1e-7)
fgrid=np.linspace(-1e-4,1e-4,301)
csvout('5.2-5.20-n3最低非零阶扫描.csv',
       ['field_atomic_units','m0_plus_linear','m0_minus_linear','m1_plus_linear_each_twofold','m1_minus_linear_each_twofold','m0_zero_quadratic','m2_zero_quadratic_each_twofold'],
       zip(fgrid,9*fgrid,-9*fgrid,4.5*fgrid,-4.5*fgrid,float(i1+i3)*fgrid**2,float(im2)*fgrid**2))
fig,axs=plt.subplots(1,2,figsize=(12.5,4.6),layout='constrained')
axs[0].plot(fgrid,9*fgrid,color='#bd5729',label='m=0，上/下斜率 ±9（各1态）')
axs[0].plot(fgrid,-9*fgrid,color='#bd5729')
axs[0].plot(fgrid,4.5*fgrid,color='#176b7a',label='|m|=1，斜率 ±9/2（各2态）')
axs[0].plot(fgrid,-4.5*fgrid,color='#176b7a')
axs[0].plot(fgrid,float(i1+i3)*fgrid**2,color='#6e5299',label='线性为零的3态（右图放大）')
axs[0].plot(fgrid,float(im2)*fgrid**2,color='#6e5299')
axs[0].set(title='9个轨道态：只保留各自最低非零阶',xlabel='电场 F / 原子单位',ylabel='能移 / Hartree')
axs[1].plot(fgrid,float(i1+i3)*fgrid**2,lw=2.2,color='#6e5299',label='m=0：−(3483/4)F²，1态')
axs[1].plot(fgrid,float(im2)*fgrid**2,lw=2.2,color='#176b7a',label='m=±2：−(1377/2)F²，2态')
axs[1].set(title='线性系数为零，能移仍从二阶开始',xlabel='电场 F / 原子单位',ylabel='能移 / Hartree')
for ax in axs:
    ax.legend(fontsize=9);ax.grid(alpha=.2);ax.ticklabel_format(style='sci',axis='both',scilimits=(-2,2))
fig.suptitle('原题 5.20：忽略精细结构与兰姆位移；自旋若保留，每个轨道态再乘2',fontsize=12.5)
savefig(fig,'5.2-5.20-n3斯塔克分裂')
summary={'source':'原题5.19、5.20：PDF379／印刷357；正文关联PDF321／印刷299',
         'lamb_samples':lamb_samples,
         'n3':{'radial_z_s_p':z01,'radial_z_p_d_m0':z12,'radial_z_p_d_m1':z12m1,
               'zero_m0_gauge_t_exact':str(t),'zero_m0_l1_E2_exact':str(i1),
               'zero_m0_l3_E2_exact':str(i3),'zero_m0_E2_exact':str(i1+i3),
               'zero_m2_E2_exact':str(im2),'zero_m0_E2_quadrature':im0_num,'zero_m2_E2_quadrature':im2_num},
         'checks':checks,'check_count':len(checks),'passed':all(c['passed'] for c in checks),
         'max_residual':max(c['residual'] for c in checks)}
(OUT/'5.2-5.19与5.20核验结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['check_count','passed','max_residual']},ensure_ascii=False))
if not summary['passed']:raise SystemExit('Verification failed')
