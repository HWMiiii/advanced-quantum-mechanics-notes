"""原题5.43—5.44。参数空间磁场采用 R>0，旋量按 sigma·Rhat 的 +1 本征值标记。
复算原理：直接求本征旋量并作离散闭环重叠，独立对连接取笛卡尔数值旋度。
输出与脚本同目录。图为教学图，未使用实验数据。"""
import cmath
import json
import math
import os
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parent.parent
deps = PROJECT/'tmp'/'section3_1_deps'
if sys.version_info[:2] == (3, 12) and deps.is_dir():
    sys.path.insert(0, str(deps))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT/'tmp'/'section5_6_phases'/'mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

PAULI = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]], dtype=complex)


def connection(point, patch):
    radius = np.linalg.norm(point)
    x, y, z = point
    rho2 = x*x+y*y
    coefficient = -(1-z/radius)/2 if patch == 'north' else (1+z/radius)/2
    return coefficient*np.array([-y,x,0])/rho2


def finite_curl(point, patch):
    delta = 1.e-5*np.linalg.norm(point)
    gradients = np.array([(connection(point+delta*axis,patch)-connection(point-delta*axis,patch))/(2*delta)
                          for axis in np.eye(3)])
    return np.array([gradients[1,2]-gradients[2,1], gradients[2,0]-gradients[0,2],
                     gradients[0,1]-gradients[1,0]])


def closed_overlap(angle, segments, spin, mu=-1., backwards=False, distort_gauge=False):
    azimuth = np.linspace(0,(-1 if backwards else 1)*2*math.pi,segments,endpoint=False)
    directions = np.column_stack([np.sin(angle)*np.cos(azimuth),np.sin(angle)*np.sin(azimuth),
                                  np.full(segments,np.cos(angle))])
    field_radius = 1.7
    hamiltonians = -mu*field_radius*np.einsum('ka,aij->kij',directions,PAULI)
    energies, eigenvectors = np.linalg.eigh(hamiltonians)
    # When mu<0 the + spin belongs to the higher energy branch.
    index = 1 if -mu*spin > 0 else 0
    vectors = eigenvectors[:,:,index]
    if distort_gauge:
        gauge_angles = np.random.default_rng(54344).uniform(-math.pi,math.pi,segments)
        vectors = vectors*np.exp(1j*gauge_angles)[:,None]
    links = np.einsum('ij,ij->i',np.roll(vectors,-1,axis=0).conj(),vectors)
    closed_product = np.prod(links/np.abs(links))
    return cmath.phase(closed_product), float(np.max(np.abs(energies[:,index]+mu*spin*field_radius)))


def phase_difference(a,b):
    return abs(cmath.phase(cmath.exp(1j*(a-b))))


def make_figure():
    for file in ('C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf'):
        if Path(file).is_file():
            font_manager.fontManager.addfont(file)
            plt.rcParams['font.family']=[font_manager.FontProperties(fname=file).get_name(),'DejaVu Sans']
            break
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(10.2,4.4),layout='constrained')
    sphere=fig.add_subplot(1,2,1,projection='3d')
    plot=fig.add_subplot(1,2,2)
    alpha=math.pi/3
    phi=np.linspace(0,2*math.pi,181)
    pp,tt=np.meshgrid(phi,np.linspace(0,alpha,32))
    sphere.plot_surface(np.sin(tt)*np.cos(pp),np.sin(tt)*np.sin(pp),np.cos(tt),
                        color='#c3dce4',alpha=.55,linewidth=0,shade=False)
    for meridian in np.linspace(0,2*math.pi,12,endpoint=False):
        theta=np.linspace(0,math.pi,100)
        sphere.plot(np.sin(theta)*np.cos(meridian),np.sin(theta)*np.sin(meridian),np.cos(theta),
                    color='#a3abae',alpha=.28,lw=.6)
    sphere.plot(np.cos(phi),np.sin(phi),np.zeros_like(phi),color='#9aa3a7',alpha=.65,lw=.9,ls='--')
    sphere.plot(np.sin(alpha)*np.cos(phi),np.sin(alpha)*np.sin(phi),np.full_like(phi,np.cos(alpha)),
                color='#246687',lw=2.3)
    p0=math.radians(-36)
    on_loop=np.array([math.sin(alpha)*math.cos(p0),math.sin(alpha)*math.sin(p0),math.cos(alpha)])
    tangent=np.array([-math.sin(p0),math.cos(p0),0])
    sphere.quiver(*on_loop,*tangent,length=.26,arrow_length_ratio=.52,color='#cc552d',linewidth=2.4)
    sphere.plot([0,on_loop[0]],[0,on_loop[1]],[0,on_loop[2]],color='#c35e33',lw=1.7)
    sphere.plot([0,0],[0,0],[0,1.24],color='#393f49',lw=1.4)
    sphere.text(0,0,1.28,'+z',fontsize=10)
    sphere.text(on_loop[0]*1.15,on_loop[1]*1.15,on_loop[2]-.08,'C',color='#245775',fontsize=12)
    arc=np.linspace(0,alpha,60)
    sphere.plot(.38*np.sin(arc)*math.cos(p0),.38*np.sin(arc)*math.sin(p0),.38*np.cos(arc),color='#775122')
    sphere.text(.22,.0,.38,r'$\vartheta$',fontsize=12)
    sphere.view_init(elev=17,azim=-62)
    sphere.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1.3),title='场方向走一周，球冠给出有向立体角')
    sphere.set_box_aspect([2,2,2.3]);sphere.set_axis_off()
    theta=np.linspace(0,math.pi,321)
    north=-(1-np.cos(theta));south=1+np.cos(theta)
    plot.plot(np.degrees(theta),north,color='#246687',lw=2.1,label='北片线积分')
    plot.plot(np.degrees(theta),south,color='#a75834',lw=2.,ls='--',label='南片线积分')
    plot.plot([60,60],[-.5,1.5],':',color='#777',lw=.9)
    plot.scatter([60,60],[-.5,1.5],s=30,c=['#246687','#a75834'],zorder=5)
    plot.text(65,.42,r'相差 $2\pi$',fontsize=10)
    plot.set(xlabel='与 +z 的夹角 ϑ（度）',ylabel=r'线积分 / $\pi$',
             xlim=(0,180),ylim=(-2.18,2.18),title='两个单值规范给出同一个相位因子')
    plot.set_xticks([0,45,90,135,180]);plot.set_yticks([-2,-1,0,1,2])
    plot.legend(loc='lower left',frameon=False,fontsize=9);plot.grid(alpha=.16)
    fig.savefig(OUTPUT/'5.6-纬圈与规范相位.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    checks=[]
    def check(label,residual,tolerance=1e-9):
        value=float(np.max(np.abs(residual)))
        checks.append({'test':label,'residual':value,'tolerance':tolerance,'passed':value<tolerance})
    for i,point in enumerate([np.array([.8,.4,1.3]),np.array([-.6,1.1,-.4])]):
        theory=-point/(2*np.linalg.norm(point)**3)
        for patch in ('north','south'):
            check(f'笛卡尔取旋度 {i} {patch}',finite_curl(point,patch)-theory,5e-9)
        x,y,z=point
        grad_phi=np.array([-y,x,0])/(x*x+y*y)
        check(f'两规范差 {i}',connection(point,'south')-connection(point,'north')-grad_phi)
    samples=[]
    for degrees in (30,60,90,120,155):
        theta=math.radians(degrees)
        for spin in (1,-1):
            for backwards in (False,True):
                theoretical=-spin*(1 if not backwards else -1)*math.pi*(1-math.cos(theta))
                numerical,e_residual=closed_overlap(theta,4096,spin,backwards=backwards)
                check(f'闭环重叠 {degrees}度 spin={spin} reverse={backwards}',phase_difference(numerical,theoretical),4e-7)
                check(f'本征能量 {degrees}度 spin={spin} reverse={backwards}',e_residual)
                samples.append({'theta_degrees':degrees,'spin':spin,'reverse':backwards,'continuum_phase_rad':theoretical,
                                'numerical_principal_phase_rad':numerical,'phase_error_mod_2pi':phase_difference(numerical,theoretical)})
    gauge_phase,_=closed_overlap(math.pi/3,4096,1,distort_gauge=True)
    clean_phase,_=closed_overlap(math.pi/3,4096,1)
    check('随机局域重相位不改变离散闭环',phase_difference(gauge_phase,clean_phase),1e-11)
    opposite_mu,_=closed_overlap(math.pi/3,4096,1,mu=1.)
    check('同一自旋方向对磁矩符号不敏感',phase_difference(opposite_mu,clean_phase),1e-11)
    nodes,weights=np.polynomial.legendre.leggauss(40)
    for degrees in (30,60,120,155):
        alpha=math.radians(degrees)
        flux=-math.pi*(alpha/2)*np.dot(weights,np.sin((nodes+1)*alpha/2))
        check(f'球冠独立数值通量 {degrees}度',flux+math.pi*(1-math.cos(alpha)))
    convergence=[]
    for count in (128,256,512,1024,2048,4096):
        value,_=closed_overlap(math.pi/3,count,1)
        convergence.append({'segments':count,'principal_phase_rad':value,'absolute_error':phase_difference(value,-math.pi/2)})
    report={'sources':'习题5.43/5.44：PDF384／印刷362；定义PDF348，能谱/曲率PDF350。',
            'angle_definition':'theta=从+z测量的真实极角；反向定义为phi由0递减到-2pi。',
            'checks':checks,'passed':all(entry['passed'] for entry in checks),'count':len(checks),
            'wilson_loop_samples':samples,'convergence_theta_60_deg':convergence}
    (OUTPUT/'5.6-几何相位原题核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'count':len(checks),'failed':[c for c in checks if not c['passed']],
                      'convergence':convergence},ensure_ascii=False))
    if not report['passed']:
        raise RuntimeError('几何相位数值核验失败')
    make_figure()


if __name__=='__main__':
    main()
