"""5.41、5.47：横向偏振投影、归一化、实空间矩阵元、末态密度与角图。"""
import json
import math
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent.parent
bundled_plot_deps = PROJECT / 'tmp' / 'section3_1_deps'
if sys.version_info[:2] == (3, 12) and bundled_plot_deps.is_dir():
    sys.path.insert(0, str(bundled_plot_deps))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / 'tmp' / 'section5_8_source' / 'mpl-exercises'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager


def angular_figure():
    for font in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
        if Path(font).exists():
            font_manager.fontManager.addfont(font)
            plt.rcParams['font.family'] = [font_manager.FontProperties(fname=font).get_name(), 'DejaVu Sans']
            break
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'font.size': 10.5})
    # b 是振子长度；u=b k_f，v=b ω/c，两图同除与角度无关的 C exp(-u²)。
    u, v = 2.0, 0.2
    theta = np.linspace(0, np.pi, 241)
    phi = np.linspace(0, 2*np.pi, 361)
    full_cut = np.sin(theta)**2 * np.exp(-v*v+2*u*v*np.cos(theta))
    intensity = full_cut[:, None] * np.cos(phi)[None, :]**2
    peak = np.arccos(2*u*v / (1+np.sqrt(1+(2*u*v)**2)))
    fig, (heat, cut) = plt.subplots(1, 2, figsize=(11.7, 4.6), layout='constrained')
    mesh = heat.pcolormesh(np.degrees(phi), np.degrees(theta), intensity, shading='auto', cmap='magma')
    heat.set(xticks=[0,90,180,270,360], yticks=[0,45,90,135,180],
             xlabel='方位角 φ（°）', ylabel='与入射方向的夹角 θ（°）', title='不同方向的相对微分截面')
    fig.colorbar(mesh, ax=heat, label='相对微分截面')
    cut.plot(np.degrees(theta), full_cut, color='#156c80', lw=2.5, label='保留光子空间相位')
    cut.plot(np.degrees(theta), np.sin(theta)**2, '--', color='#a44d42', lw=1.8, label='长波近似')
    cut.axvline(np.degrees(peak), color='#666666', lw=1, ls=':')
    cut.text(np.degrees(peak)-2, 0.21, f'峰值 {np.degrees(peak):.1f}°', ha='right', color='#444444')
    cut.set(xlim=(0,180), ylim=(0,full_cut.max()*1.18), xticks=[0,45,90,135,180],
            xlabel='与入射方向的夹角 θ（°）', ylabel='相对微分截面', title='偏振平面内的截面（φ = 0°）')
    cut.legend(loc='upper right', frameon=False, fontsize=9)
    fig.suptitle('原题5.47　线偏振光照射高斯轨道  ·  b k$_f$ = 2.0，b ω/c = 0.2', fontsize=14)
    fig.savefig(OUT/'5.8-振子光电截面角分布.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    checks = []
    def verify(name, difference, tolerance=1e-11):
        error = float(np.max(np.abs(difference)))
        checks.append({'name': name, 'max_abs_difference': error,
                       'tolerance': tolerance, 'passed': error <= tolerance})

    # 5.41：直接从笛卡尔偶极投影到局部横向平面，再独立投到两个螺旋度。
    dipole = np.array([1, 1j, 0], complex)/np.sqrt(2)
    patterns = []
    for theta, phi in [(0,0), (np.pi/4,0.3), (np.pi/2,1.1), (3*np.pi/4,1.6), (np.pi,0)]:
        c, s = np.cos(theta),np.sin(theta)
        ray = np.array([s*np.cos(phi),s*np.sin(phi),c])
        polar = np.array([c*np.cos(phi),c*np.sin(phi),-s])
        azimuth = np.array([-np.sin(phi),np.cos(phi),0])
        field = dipole-ray*np.vdot(ray,dipole)
        verify(f'5.41 θ={theta:.4g} 横向性', np.vdot(ray,field))
        jones = np.array([np.vdot(polar,field),np.vdot(azimuth,field)])
        verify(f'5.41 θ={theta:.4g} 复偏振分量', jones-np.exp(1j*phi)/np.sqrt(2)*np.array([c,1j]))
        weights = [abs(np.vdot((polar+1j*h*azimuth)/np.sqrt(2),field))**2 for h in (+1,-1)]
        verify(f'5.41 θ={theta:.4g} 两个螺旋度强度', np.array(weights)-np.array([(1+c)**2,(1-c)**2])/4)
        patterns.append({'theta_rad':float(theta),'helicity_weights':weights})
    us, ws = np.polynomial.legendre.leggauss(100)
    verify('5.41 归一化角概率', 2*np.pi * np.sum(ws*3/(16*np.pi)*(1+us*us))-1)

    # 5.47：独立积每个坐标的原始 |k> 与 p_x ψ 乘积，不使用分部积分和高斯变换闭式。
    b, kf, wave_k, omega = 0.8, 1.7, 0.2, 3.2
    photon_speed, omega0 = omega / wave_k, b**-2  # m_e=ℏ=1
    alpha=1/137
    xgrid, weights = 10*b*us, 10*b*ws
    gaussian = np.exp(-xgrid*xgrid/(2*b*b))
    def fourier_part(component, derivative):
        operator = 1j*xgrid/b**2 if derivative else 1
        return np.sum(weights * np.exp(-1j*component*xgrid) * operator * gaussian)

    differential = []
    for theta,phi in [(0,0),(0.4,0.2),(np.pi/2,0),(1.9,0.8),(np.pi/2,np.pi/2),(np.pi,0)]:
        wavevector = kf*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        q = wavevector-np.array([0,0,wave_k])
        raw = (np.pi*b*b)**(-.75) * np.prod([fourier_part(val,j==0) for j,val in enumerate(q)])
        analytic = wavevector[0]*2**1.5*np.pi**.75*b**1.5*np.exp(-b*b*np.dot(q,q)/2)
        verify(f'5.47 θ={theta:.4g},φ={phi:.4g} 原始动量矩阵元', raw-analytic)
        # 使用印刷公式分离写出的前因子及两个指数；与不同盒长的(5.338)逐一对照。
        prefactor = 4*alpha*kf**3/(omega*omega0)*np.sqrt(np.pi/omega0)
        printed = (prefactor*np.exp(-(kf*kf+wave_k**2)/omega0)
                   *np.sin(theta)**2*np.cos(phi)**2*np.exp(2*kf*omega*np.cos(theta)/(omega0*photon_speed)))
        for box in (13, 27):
            cross_section=(4*np.pi**2*alpha/omega)*abs(raw/box**1.5)**2*kf*box**3/(2*np.pi)**3
            verify(f'5.47 θ={theta:.4g},φ={phi:.4g},L={box} 截面及盒长消去',cross_section-printed)
        differential.append({'theta_rad':float(theta),'phi_rad':float(phi),'cross_section':float(printed)})
    chi=0.8
    quad = np.pi*np.sum(ws*(1-us*us)*np.exp(chi*us))
    integral=4*np.pi*(chi*np.cosh(chi)-np.sinh(chi))/chi**3
    verify('5.47 全立体角积分',quad-integral)
    angular_figure()
    report={'5.41':patterns,'5.47':{'units':'m_e=hbar=1; illustrative parameters',
            'b':b,'kf':kf,'incident_wavenumber':wave_k,'omega':omega,'c':photon_speed,
            'differential':differential},'checks':checks,'all_passed':all(x['passed'] for x in checks)}
    (OUT/'5.8-圆偶极与振子光电截面核验.json').write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f"{sum(x['passed'] for x in checks)}/{len(checks)} checks passed")
    return 0 if report['all_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
