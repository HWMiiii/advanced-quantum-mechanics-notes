"""Section 3.1: exact rotation checks and a teaching figure, no external data.

Requires numpy and matplotlib. Run this file to reproduce the adjacent PNG,
SVG and JSON. All angles are in radians; matrices act on column vectors.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path(__file__).resolve().parent

def rx(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[1.,0,0],[0,c,-s],[0,s,c]])

def ry(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1.,0],[-s,0,c]])

def rz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

def main():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.size']=11
    e=np.logspace(-3,-.6,70)
    signal=[]; error=[]
    for a in e:
        C=rx(a)@ry(a)@rx(-a)@ry(-a)
        signal.append(np.linalg.norm(C-np.eye(3)))
        error.append(np.linalg.norm(C-rz(a*a)))
    signal=np.array(signal); error=np.array(error)
    # Fit only the small-angle portion, above the floating-point floor.
    sig_slope=float(np.polyfit(np.log(e[:30]),np.log(signal[:30]),1)[0])
    err_slope=float(np.polyfit(np.log(e[:30]),np.log(error[:30]),1)[0])
    assert abs(sig_slope-2)<1e-3
    assert abs(err_slope-3)<1e-3
    initial=np.array([1.,0,0])
    zx=rx(np.pi/2)@rz(np.pi/2)@initial
    xz=rz(np.pi/2)@rx(np.pi/2)@initial
    assert np.allclose(zx,[0,0,1]) and np.allclose(xz,[0,1,0])

    fig=plt.figure(figsize=(12.5,5.2),facecolor='#f9fafc')
    ax=fig.add_subplot(121,projection='3d',facecolor='#f9fafc')
    colors=['#8c98a8','#b14935','#246c9d']
    for v,label in zip(np.eye(3),['x','y','z']):
        ax.quiver(0,0,0,*v,color='#8994a2',arrow_length_ratio=.08,linewidth=1)
        ax.text(*(v*1.10),label,color='#4b5563')
    theta=np.linspace(0,np.pi/2,100)
    path_z=np.array([rz(t)@initial for t in theta])
    path_x=np.array([rx(t)@np.array([0.,1,0]) for t in theta])
    ax.plot(*path_z.T,color=colors[1],lw=3,label='先 z 后 x：x → y → z')
    ax.plot(*path_x.T,color=colors[1],lw=3)
    ax.plot(*(.96*path_z).T,color=colors[2],lw=2,ls='--',label='先 x 后 z：x → x → y')
    for v,c in [(initial,colors[0]),(zx,colors[1]),(xz,colors[2])]:
        ax.scatter(*v,color=c,s=55,depthshade=False)
    ax.set(xlim=(-.1,1.2),ylim=(-.1,1.2),zlim=(-.1,1.2))
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=24,azim=34)
    ax.set_axis_off()
    ax.set_title('两次 90° 主动旋转：次序改变终点',pad=12)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.05),frameon=False,fontsize=10)
    ax2=fig.add_subplot(122,facecolor='#f9fafc')
    ax2.loglog(e,signal,color=colors[1],lw=2.3,label=r'剩余信号 $\|C-I\|_F$：二阶')
    ax2.loglog(e,error,color=colors[2],lw=2.3,label=r'近似误差 $\|C-R_z(\epsilon^2)\|_F$：三阶')
    ax2.set_xlabel(r'每步小角 $\epsilon$ / rad')
    ax2.set_ylabel('Frobenius 矩阵范数')
    ax2.set_title('四次小旋转：二阶主项可被分辨',pad=12)
    ax2.grid(True,which='both',alpha=.18)
    ax2.legend(loc='upper left',fontsize=9,frameon=False)
    ax2.text(.97,.05,f'小角拟合斜率：{sig_slope:.4f} / {err_slope:.4f}',transform=ax2.transAxes,ha='right',fontsize=10)
    fig.suptitle('§3.1 旋转次序与二阶非交换性',fontsize=16,y=.98)
    fig.subplots_adjust(top=.83,bottom=.16,left=.03,right=.98,wspace=.12)
    png=OUT/'3.1-旋转次序与二阶效应.png'
    fig.savefig(png,dpi=180)
    fig.savefig(png.with_suffix('.svg'))
    plt.close(fig)
    data={'convention':'active; rightmost matrix acts first; radians',
          'C':'Rx(e) Ry(e) Rx(-e) Ry(-e)',
          'signal_small_angle_slope':sig_slope,'error_small_angle_slope':err_slope,
          'start_vector':initial.tolist(),'z_then_x':zx.tolist(),'x_then_z':xz.tolist(),
          'samples':[{'epsilon':float(a),'signal':float(b),'approximation_error':float(c)} for a,b,c in zip(e,signal,error)]}
    (OUT/'3.1-旋转核验结果.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k!='samples'},ensure_ascii=False))

if __name__=='__main__':
    main()
