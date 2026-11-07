"""原题5.11/16/17/21/22/23的独立矩阵、积分及选择定则核验。"""
from pathlib import Path
from fractions import Fraction as F
import math,json
import numpy as np
OUT=Path(__file__).resolve().parent
checks=[]
def check(name,actual,expected,tol=1e-10):
    err=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    checks.append(dict(name=name,residual=err,tolerance=tol,passed=err<tol))
sz=np.diag([.5,-.5]); sx=np.array([[0,1],[1,0]])/2; sy=np.array([[0,-1j],[1j,0]])/2
singlet=np.array([0,1,-1,0])/np.sqrt(2);triplet=np.array([0,1,1,0])/np.sqrt(2)
Dz=np.kron(sz,np.eye(2))-np.kron(np.eye(2),sz)
check('5.11(d)电子偶素单三重态矩阵元',singlet@Dz@triplet,1)
Sdot=sum(np.kron(s,s) for s in [sx,sy,sz])
check('5.11(e)电子单态自旋关联',singlet@Sdot@singlet,-.75)
check('5.11(c)CG权重求Lz',8/9*3+1/9*4,28/9)

x,w=np.polynomial.legendre.leggauss(120);t=(x+1)/2;wt=w/2
finite=[]
for R in [.001,.002]:
    r=R*t;kernel=t-1.5*t*t+.5*t**4
    radial=[2*np.exp(-r),(2-r)*np.exp(-r/2)/(2*np.sqrt(2)),r*np.exp(-r/2)/(2*np.sqrt(6))]
    exact=np.array([R*R*np.sum(wt*kernel*f*f) for f in radial])
    lead=np.array([2*R*R/5,R*R/20,R**4/1120])
    check(f'5.16 有限核半径主项相对误差 R/a0={R}',exact/lead,np.ones(3),2*R)
    finite.append(dict(R_over_a0=R,first_order_integrals=exact.tolist(),leading_terms=lead.tolist()))
    assert np.all(exact>0)
check('5.16 小核极限2p/2s比值系数',F(1,1120)/F(1,20),F(1,56))

for n in range(1,6):
    for l in range(n):
        K=F(3,4*n)-F(2,2*l+1)
        if l==0:check(f'5.17 n={n},l=0 Darwin加动能',float(K+1),float(F(3,4*n)-1))
        else:
            for sign in [-1,1]:
                so=F(1,(2*l+1)*(l+1)) if sign==1 else -F(1,l*(2*l+1))
                denom=l+1 if sign==1 else l
                check(f'5.17 n={n},l={l},j=l{sign:+}/2',float(K+so),float(F(3,4*n)-F(1,denom)))

theta=.71;phi=.43
nvec=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
sigmar=2*(nvec[0]*sx+nvec[1]*sy+nvec[2]*sz)
splus=np.array([1,0])/np.sqrt(4*np.pi)
Y10=np.sqrt(3/(4*np.pi))*np.cos(theta)
Y11=-np.sqrt(3/(8*np.pi))*np.sin(theta)*np.exp(1j*phi)
pplus=np.array([-Y10/np.sqrt(3),np.sqrt(2/3)*Y11])
check('5.21 sigma.rhat将s1/2映为负p1/2',sigmar@splus,-pplus)
check('5.22 三维振子原点密度',1/(2*np.pi)*2/np.sqrt(np.pi),np.pi**(-1.5))
check('5.22 氢1s原点密度',1/(2*np.pi)*2,1/np.pi)

l=3; mvals=np.arange(-l,l+1); lp=np.zeros((2*l+1,2*l+1),complex)
for i,m in enumerate(mvals[:-1]):lp[i+1,i]=np.sqrt(l*(l+1)-m*(m+1))
ly=(lp-lp.conj().T)/(2j);B=2.;C=.01
H=B*np.diag(mvals)+C*ly
check('5.23(a)旋转磁场精确能谱',np.linalg.eigvalsh(H),mvals*np.sqrt(B*B+C*C))
check('5.23(a)二阶谱余项',np.linalg.eigvalsh(H),mvals*(B+C*C/(2*B)),l*C**4/B**3)

# Independently integrate spherical harmonics to check rank-2 selection rules.
xx,ww=np.polynomial.legendre.leggauss(32);pp=np.arange(64)*2*np.pi/64
X,P=np.meshgrid(xx,pp,indexing='ij');weight=np.repeat(ww,64)*2*np.pi/64
def Y(l,m):
    if m<0:return (-1)**(-m)*np.conj(Y(l,-m))
    assoc=(-1)**m*(1-X*X)**(m/2)*np.polynomial.legendre.Legendre.basis(l).deriv(m)(X)
    norm=math.sqrt((2*l+1)/(4*np.pi)*math.factorial(l-m)/math.factorial(l+m))
    return norm*assoc*np.exp(1j*m*P)
states=[(l,m) for l in range(5) for m in range(-l,l+1)]
YY=np.array([Y(l,m).ravel() for l,m in states])
for label,op,dm in [('3z2-r2',3*X*X-1,{0}),('xy',(1-X*X)*np.sin(P)*np.cos(P),{-2,2})]:
    mat=(YY.conj()*(weight*op.ravel()))@YY.T
    forbidden=[]
    for i,(ll,mm) in enumerate(states):
        for j,(l,m) in enumerate(states):
            allowed=abs(ll-l)<=2<=ll+l and (ll+l)%2==0 and mm-m in dm
            if not allowed:forbidden.append(mat[i,j])
    check('5.23(b)角积分全部禁戒矩阵元 '+label,forbidden,np.zeros(len(forbidden)),2e-13)

res=dict(checks=checks,finite_nucleus=finite,all_passed=all(c['passed'] for c in checks))
(OUT/'5.3-径向与选择定则原题核验.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
if not res['all_passed']:raise SystemExit('verification failed')
