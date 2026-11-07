"""§5.1 所选原题的教学数值核验；不是原题额外要求，不代表个人完成。"""
from pathlib import Path
import json
import math

OUT = Path(__file__).resolve().parent
import numpy as np

checks = []
def check(name, actual, expected, tolerance=1e-10):
    residual = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    checks.append(dict(name=name, residual=residual, tolerance=tolerance,
                       passed=residual < tolerance))

N = 60
a = np.diag(np.sqrt(np.arange(1, N)), 1)
x = (a+a.T)/math.sqrt(2)
p = 1j*(a.T-a)/math.sqrt(2)
H0 = np.diag(np.arange(N)+0.5)
b = .1
check("5.1 线性微扰振子基态与精确平方配方", np.linalg.eigvalsh(H0+b*x)[0], .5-b*b/2)
check("5.3 振子基态p四次方矩", (p@p@p@p)[0,0].real, .75)
check("5.8 x平方作用于基态的全部非零分量", (x@x)[:5,0], [.5,0,math.sqrt(2)/2,0,0])

def rotor(F, size):
    # A=1; cos basis: 1/sqrt(2pi), cos(n phi)/sqrt(pi); sin basis: sin(n phi)/sqrt(pi).
    hc=np.diag(np.arange(size,dtype=float)**2)
    hs=np.diag(np.arange(1,size+1,dtype=float)**2)
    for i in range(size-1):
        hc[i,i+1]=hc[i+1,i]=-F/(math.sqrt(2) if i==0 else 2)
        hs[i,i+1]=hs[i+1,i]=-F/2
    return np.linalg.eigvalsh(hc),np.linalg.eigvalsh(hs)

rotor_rows=[]
for F in [.02,.01]:
    ec,es=rotor(F,40)
    ec60,es60=rotor(F,60)
    check(f"5.4 转子截断收敛 F={F}", np.r_[ec[:5],es[:4]], np.r_[ec60[:5],es60[:4]],1e-11)
    predictions=np.array([-F*F/2,1+5*F*F/12,1-F*F/12,4+F*F/30,4+F*F/30])
    actual=np.array([ec[0],ec[1],es[0],ec[2],es[1]])
    check(f"5.4 五个低态二阶余项为O(F^4), F={F}", actual, predictions, F**4)
    rotor_rows.append(dict(F=F, labels=["m=0","cos(phi)","sin(phi)","cos(2phi)","sin(2phi)"],
                           exact=actual.tolist(), second_order=predictions.tolist(),
                           residual=(actual-predictions).tolist()))

delta=2.; g=.02
h=np.array([[delta/2,g],[g,-delta/2]])
R=math.sqrt((delta/2)**2+g*g)
theta=.5*math.atan2(2*g,delta)
v=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
check("5.14 精确本征矢与本征值", h@v, v@np.diag([R,-R]))
check("5.14 精确本征矢正交归一", v.T@v,np.eye(2))
check("5.14 非简并能量展开四阶余项", R, delta/2+g*g/delta, 2*g**4/abs(delta)**3)
overlap=math.cos(theta)**2
check("5.5 原态概率二阶展开",overlap,1-(g/delta)**2,4*(g/delta)**4)

result=dict(description="教学参数核验，不是原题指定数值任务；无教材PDF修改", checks=checks,
            rotor=rotor_rows, all_passed=all(item['passed'] for item in checks))
(OUT/'5.1-原题解析结果核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
if not result['all_passed']: raise SystemExit('verification failed')
