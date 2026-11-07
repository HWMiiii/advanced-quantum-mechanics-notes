"""习题5.45、5.46、5.48。数值坐标采用 a0=hbar=m=1。
从实空间波函数分别核验三维傅里叶变换、平面波矩阵元、能壳角积分和归一化。
不使用实验数据。输出图、JSON与本脚本同目录。"""
import json
import math
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent.parent
deps = PROJECT/'tmp'/'section3_1_deps'
if sys.version_info[:2] == (3, 12) and deps.is_dir():
    sys.path.insert(0, str(deps))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT/'tmp'/'section5_7_continuum'/'mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

PI=math.pi
nodes, weights = np.polynomial.legendre.leggauss(120)


def momentum_amplitude(q):
    return 2*math.sqrt(2)/PI/(1+q*q)**2


def direct_fourier(q):
    # Integrate r and cos(theta) separately; choose Fourier axis along q.
    r=23*(nodes+1)
    angular=2*PI*(np.exp(-1j*q*r[:,None]*nodes[None,:])@weights)
    return 23*np.sum(weights*r*r*np.exp(-r)*angular)/(math.sqrt(PI)*(2*PI)**1.5)


def angular_density(u,v,cosine):
    return (1+u*u+v*v-2*u*v*cosine)**(-4)


def numerical_spatial_matrix(u,v,theta):
    # z remains along the incident wave; momentum is in the xz plane.
    # Integrate exp[-i p.r + ik z] without first rotating to p-k.
    r=23*(nodes+1)
    polar_nodes, polar_weights = np.polynomial.legendre.leggauss(58)
    azimuth=2*PI*np.arange(96)/96
    nz=polar_nodes[:,None]
    nx=np.sqrt(1-nz*nz)*np.cos(azimuth)[None,:]
    projection=u*math.sin(theta)*nx+(u*math.cos(theta)-v)*nz
    angle_sums=np.array([np.sum(polar_weights[:,None]*np.exp(-1j*radius*projection))*
                         (2*PI/len(azimuth)) for radius in r])
    return 23*np.sum(weights*r**2*np.exp(-r)*angle_sums)/(math.sqrt(PI)*(2*PI)**1.5)


def figure():
    for candidate in ('C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf'):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams['font.family']=[font_manager.FontProperties(fname=candidate).get_name(),'DejaVu Sans']
            break
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'font.size':10})
    fig,(ang,rad)=plt.subplots(1,2,figsize=(10.4,3.9),layout='constrained')
    theta=np.linspace(0,PI,301)
    u,v=1.,.3
    scalar=angular_density(u,v,np.cos(theta))
    photon_v=(1+u*u)/(2*137.036)
    photon_form=angular_density(u,photon_v,np.cos(theta))
    transverse=np.sin(theta)**2*photon_form
    ang.plot(np.degrees(theta),scalar/np.max(scalar),color='#204e86',lw=2,label='标量势，ka₀=0.30')
    ang.plot(np.degrees(theta),photon_form/np.max(photon_form),color='#858a8e',lw=1.7,ls=':',
             label='标量势，ka₀≈1/137')
    ang.plot(np.degrees(theta),transverse/np.max(transverse),color='#b25730',ls='--',lw=2,
             label='x 偏振光，ka₀≈1/137，φ=0')
    ang.set(xlabel='出射方向与 +z 的夹角 θ（度）',ylabel='相对角强度（各曲线峰值归一）',
            title='同一形状因子，耦合算符不同',xlim=(0,180),ylim=(0,1.06))
    ang.set_xticks([0,45,90,135,180])
    handles,labels=ang.get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    ugrid=np.linspace(0,3.5,400)
    probability=(32/PI)*ugrid**2/(1+ugrid**2)**4
    rad.plot(ugrid,probability,color='#21735d',lw=2)
    peak=1/math.sqrt(3)
    rad.axvline(peak,color='#ad632b',lw=1.1,ls=':')
    rad.text(peak+.08,0.22,'峰位 1/√3',color='#8e552a')
    rad.set(xlabel='动量大小 u = a₀p / ℏ',ylabel='每单位 u 的概率密度',
            title='氢 1s 的径向动量分布',xlim=(0,3.5),ylim=(0,None))
    for ax in (ang,rad):
        ax.grid(alpha=.15)
    fig.savefig(OUT/'5.7-标量行波角分布与氢动量.png',dpi=190,bbox_inches='tight')
    plt.close(fig)


def main():
    checks=[]
    def check(label,value,tolerance=1e-9):
        residual=float(abs(value))
        checks.append({'test':label,'residual':residual,'tolerance':tolerance,'passed':residual<tolerance})
    for q in [0,.3,.8,1.5,2.5,3.]:
        check(f'5.48 r、theta独立积分，q={q}',direct_fourier(q)-momentum_amplitude(q))
    t=(nodes+1)*PI/4
    radial_normalization=float((PI/4)*np.dot(weights,32/PI*np.sin(t)**2*np.cos(t)**4))
    check('5.48 全动量空间概率归一',radial_normalization-1)
    u,v=1.,.3
    angular_samples=[]
    for degrees in [0,45,90,135,180]:
        theta=math.radians(degrees)
        q=math.sqrt((u*math.sin(theta))**2+(u*math.cos(theta)-v)**2)
        numerical=numerical_spatial_matrix(u,v,theta)
        analytic=momentum_amplitude(q)
        check(f'5.45 入射轴固定的三维积分 theta={degrees}',numerical-analytic)
        angular_samples.append({'degrees':degrees,'p_minus_k':q,'analytic_delta_normalized_overlap':analytic,
                                'direct_overlap_real':float(numerical.real),'direct_overlap_imag':float(numerical.imag)})
    A=1+u*u+v*v
    B=2*u*v
    angular_integral=2*PI*np.dot(weights,angular_density(u,v,nodes))
    angular_exact=2*PI*((A-B)**(-3)-(A+B)**(-3))/(3*B)
    check('5.45 能壳角分布积分',angular_integral-angular_exact)
    q=math.sqrt(u*u+v*v)
    for length in [50.,100.]:
        mbox=(2*PI)**1.5*momentum_amplitude(q)/length**1.5
        check(f'5.45 L={length:g} 箱规因子抵消',mbox**2*(length/(2*PI))**3-momentum_amplitude(q)**2)
    for n in [10,100,1000]:
        length=2.1
        const=PI**2/(2*length**2)
        midpoint_e=const*(n+.5)**2
        spacing=const*((n+1)**2-n**2)
        rho=length/(PI*math.sqrt(2*midpoint_e))
        check(f'5.46 n={n} 局部能级计数',rho*spacing-1)
    report={'source':'5.45 PDF384—385；5.46/5.48 PDF385。无量纲数值：a0=hbar=m=1。',
            'checks':checks,'passed':all(c['passed'] for c in checks),'count':len(checks),
            'radial_normalization':radial_normalization,'figure_parameters':{'u':u,'scalar_v':v,'photon_v':(1+u*u)/(2*137.036)},
            'angular_samples':angular_samples,'angular_integral':float(angular_integral)}
    (OUT/'5.7-连续谱与动量分布核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'count':len(checks),'passed':report['passed'],'failed':[c for c in checks if not c['passed']],
                      'radial_norm':radial_normalization,'angular_integral':angular_integral},ensure_ascii=False))
    if not report['passed']:
        raise RuntimeError('积分核验失败')
    figure()


if __name__=='__main__':
    main()
