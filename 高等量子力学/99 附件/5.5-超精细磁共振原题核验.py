"""教材5.39和5.40，采用 ℏ=1，以无扰超精细能隙为单位。
从两个独立的自旋1/2算符构造四维矩阵。核对能量、本征矢、纵横向选择定则与弱场展开。
图和参数扫描的位置与本脚本相同。"""
import csv
import json
import math
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent.parent
deps = PROJECT / 'tmp' / 'section3_1_deps'
if sys.version_info[:2] == (3, 12) and deps.is_dir():
    sys.path.insert(0, str(deps))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / 'tmp' / 'section5_5_magnetic' / 'mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

SIGMA = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]
S1 = [np.kron(s/2, np.eye(2)) for s in SIGMA]
S2 = [np.kron(np.eye(2), s/2) for s in SIGMA]
EXCHANGE = sum(a @ b for a, b in zip(S1, S2))
TPLUS, TMINUS = np.eye(4)[0], np.eye(4)[3]
TZERO = np.array([0, 1, 1, 0])/math.sqrt(2)
SINGLET = np.array([0, 1, -1, 0])/math.sqrt(2)
TRIPLET = np.column_stack([TPLUS, TZERO, TMINUS])


def model(proton_coefficient):
    # First spin couples with coefficient 1. The second coefficient is -1 for
    # positronium and 0 for hydrogen with the electron-only Zeeman interaction.
    return [a + proton_coefficient*b for a, b in zip(S1, S2)]


def result(delta, b, second):
    mixing = b*(1-second)/2
    theta = .5*math.atan(2*mixing/delta)
    gap = np.hypot(delta, 2*mixing)
    energies = np.array([delta/4+b*(1+second)/2, delta/4-b*(1+second)/2,
                         -delta/4+math.copysign(gap/2, delta), -delta/4-math.copysign(gap/2, delta)])
    t = math.cos(theta)*TZERO + math.sin(theta)*SINGLET
    s = -math.sin(theta)*TZERO + math.cos(theta)*SINGLET
    return energies, np.column_stack([TPLUS,TMINUS,t,s]), theta


def magnitude(values):
    return float(np.max(np.abs(values)))


def verify(second, label):
    operators = model(second)
    records = []
    def record(name, value, tolerance=2e-12):
        residual = magnitude(value)
        records.append({'test':name, 'residual':residual, 'tolerance':tolerance, 'passed':residual < tolerance})
    factor=(1-second)/2
    record(f'{label} 三重态一阶投影', TRIPLET.conj().T @ operators[2] @ TRIPLET -
           np.diag([(1+second)/2,0,-(1+second)/2]))
    record(f'{label} 零场态连接系数', SINGLET @ operators[2] @ TZERO - factor)
    errors = {name: [] for name in ['energy','eigenvector','orthonormality','longitudinal','transverse']}
    for delta in (1.,-.7):
        for b in (-1.,-.2,0.,.11,.8):
            h = delta*EXCHANGE+b*operators[2]
            energies, vectors, theta = result(delta,b,second)
            errors['energy'].append(magnitude(np.linalg.eigvalsh(h)-np.sort(energies)))
            errors['eigenvector'].append(magnitude(h@vectors - vectors*energies[np.newaxis,:]))
            errors['orthonormality'].append(magnitude(vectors.conj().T @ vectors-np.eye(4)))
            t,s = vectors[:,2], vectors[:,3]
            errors['longitudinal'].append(magnitude(np.vdot(s, operators[2] @ t)-factor*math.cos(2*theta)))
            errors['transverse'].extend(magnitude(np.vdot(s, op @ t)) for op in operators[:2])
    for name,values in errors.items():
        record(f'{label} 完整4维核验 {name}', values)
    h=1.e-4
    e0,v0,_=result(1.,0.,second)
    eplus,vplus,_=result(1.,h,second)
    eminus,vminus,_=result(1.,-h,second)
    record(f'{label} 二阶能量系数', (eplus+eminus-2*e0)/(2*h*h)-np.array([0,0,factor**2,-factor**2]), 6e-8)
    first_derivative = np.column_stack([0*TPLUS,0*TMINUS,factor*SINGLET,-factor*TZERO])
    record(f'{label} 一阶态系数', (vplus-vminus)/(2*h)-first_derivative, 3e-8)
    return records


def graph(rows):
    for candidate in ('C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf'):
        if Path(candidate).is_file():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams['font.family'] = [font_manager.FontProperties(fname=candidate).get_name(),'DejaVu Sans']
            break
    plt.rcParams.update({'axes.spines.top':False, 'axes.spines.right':False,'axes.unicode_minus':False,
                         'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.0),layout='constrained',sharey=True)
    for ax,label in zip(axes,['5.39','5.40']):
        points=[r for r in rows if r['exercise']==label]
        x=[r['b_over_Delta'] for r in points]
        for key,legend,color,ls in [('E_T0','接三重态 m=0','#286299','-'),
                                    ('E_S','接单态 m=0','#a94441','-'),
                                    ('E_Tplus','三重态 m=+1','#387951','--'),
                                    ('E_Tminus','三重态 m=−1','#a66d1f',':')]:
            if label=='5.39' and key=='E_Tminus': continue
            if label=='5.39' and key=='E_Tplus': legend='三重态 m=±1（重合）'
            ax.plot(x,[r[key] for r in points],color=color,ls=ls,lw=2,label=legend)
        ax.set(xlabel='磁场能量 b / Δ',xlim=(-1.1,1.1),ylim=(-1.75,1.05))
        ax.axvline(0,color='#69717a',lw=.6)
        ax.grid(alpha=.17)
        ax.legend(loc='lower center',ncol=2,fontsize=8,frameon=True)
    axes[0].set(ylabel='能量 E / Δ',title='5.39　电子偶素')
    axes[1].set(title='5.40　氢原子（题给电子塞曼项）')
    fig.savefig(OUT/'5.5-超精细四态随磁场变化.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    checks = verify(-1,'5.39') + verify(0,'5.40')
    rows=[]
    for label,second in [('5.39',-1),('5.40',0)]:
        for x in np.linspace(-1.1,1.1,221):
            energies,vectors,theta=result(1.,x,second)
            rows.append(dict(exercise=label,b_over_Delta=float(x),E_Tplus=energies[0],
                    E_Tminus=energies[1],E_T0=energies[2],E_S=energies[3],
                    angular_frequency_times_hbar_over_Delta=energies[2]-energies[3],
                    longitudinal_spin_element_over_hbar=(1-second)*math.cos(2*theta)/2))
    with (OUT/'5.5-超精细四态参数扫描.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report={'source':'5.39 PDF383—384，5.40 PDF384；独立积态数值矩阵，默认 hbar=1。',
            'sample_b_over_Delta_0p1':[r for r in rows if abs(r['b_over_Delta']-.1)<1e-12],
            'count':len(checks),'passed':all(row['passed'] for row in checks),'checks':checks}
    (OUT/'5.5-超精细磁共振原题核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'checks':len(checks),'examples':report['sample_b_over_Delta_0p1'],
                      'failed':[row for row in checks if not row['passed']]},ensure_ascii=False))
    if not report['passed']:
        raise RuntimeError('磁共振矩阵核验失败')
    graph(rows)


if __name__=='__main__':
    main()
