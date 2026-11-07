"""Section 3.3 teaching figure and matrix checks; numpy and matplotlib required.

Angles are radians. Active rotations act on column vectors. No experimental
data or textbook images are used.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path(__file__).resolve().parent
I2=np.eye(2,dtype=complex)
I3=np.eye(3)
SIG=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],dtype=complex)
X,Y,Z=I3

def nsigma(n):
    return np.einsum('i,ijk->jk',n,SIG)

def U(n,t):
    return np.cos(t/2)*I2-1j*np.sin(t/2)*nsigma(n)

def R(n,t):
    n=np.asarray(n)
    cross=np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    return np.cos(t)*I3+(1-np.cos(t))*np.outer(n,n)+np.sin(t)*cross

def lift_to_R(u):
    return np.array([[np.trace(SIG[i]@u@SIG[j]@u.conj().T).real/2
                      for j in range(3)] for i in range(3)])

def CK(a,b):
    return np.array([[a,b],[-b.conjugate(),a.conjugate()]])

def euler_closed(a,b,g):
    c,s=np.cos(b/2),np.sin(b/2)
    return np.array([[c*np.exp(-1j*(a+g)/2),-s*np.exp(-1j*(a-g)/2)],
                     [s*np.exp(1j*(a-g)/2),c*np.exp(1j*(a+g)/2)]])

def main():
    rng=np.random.default_rng(33026)
    errors={}
    def check(key,value):
        errors[key]=max(errors.get(key,0.),float(np.max(np.abs(value))))
    for _ in range(100):
        n=rng.normal(size=3);n/=np.linalg.norm(n)
        m=rng.normal(size=3);m/=np.linalg.norm(m)
        t,s=rng.uniform(-4*np.pi,4*np.pi,2)
        u,v=U(n,t),U(m,s)
        r=R(n,t)
        check('SO3_orthogonality',r.T@r-I3)
        check('SO3_determinant',np.linalg.det(r)-1)
        check('SU2_unitarity',u.conj().T@u-I2)
        check('SU2_determinant',np.linalg.det(u)-1)
        check('SU2_to_SO3',lift_to_R(u)-r)
        check('double_cover',lift_to_R(-u)-r)
        check('homomorphism',lift_to_R(u@v)-lift_to_R(u)@lift_to_R(v))
        a1,b1=u[0];a2,b2=v[0]
        check('CK_product',CK(a1*a2-b1*b2.conjugate(),a1*b2+b1*a2.conjugate())-u@v)
        check('CK_inverse',CK(a1.conjugate(),-b1)-u.conj().T)
        a,g=rng.uniform(-3*np.pi,3*np.pi,2);b=rng.uniform(.05,np.pi-.05)
        rz,ry,rg=R(Z,a),R(Y,b),R(Z,g)
        moving_y=R(rz@Y,b)
        q2=moving_y@rz
        moving_z=R(q2@Z,g)
        total=moving_z@q2
        expected=rz@ry@rg
        check('moving_y_conjugation',moving_y-rz@ry@rz.T)
        check('moving_z_conjugation',moving_z-moving_y@rg@moving_y.T)
        check('intrinsic_extrinsic_equivalence',total-expected)
        eu=U(Z,a)@U(Y,b)@U(Z,g)
        check('Euler_four_entries',eu-euler_closed(a,b,g))
        check('Euler_R_from_U',lift_to_R(eu)-expected)
        br=np.arccos(np.clip(expected[2,2],-1,1))
        ar=np.arctan2(expected[1,2],expected[0,2])
        gr=np.arctan2(expected[2,1],-expected[2,0])
        check('R_Euler_reconstruction',R(Z,ar)@R(Y,br)@R(Z,gr)-expected)
        ca,cb=eu[0];bu=2*np.arctan2(abs(cb),abs(ca))
        p,q=-np.angle(ca),-np.angle(-cb)
        check('CK_Euler_reconstruction',euler_closed(p+q,bu,p-q)-eu)
        check('pole_beta_zero',euler_closed(a,0,g)-U(Z,a+g))
        check('pole_beta_pi',euler_closed(a,np.pi,g)-U(Z,a-g)@U(Y,np.pi))
    sampleR=R(Z,np.pi/2)@R(Y,np.pi/2)
    sampleU=U(Z,np.pi/2)@U(Y,np.pi/2)
    sample_n=np.array([-1.,1,1])/np.sqrt(3)
    check('worked_example_axis_angle',sampleU-U(sample_n,2*np.pi/3))
    check('worked_example_R',sampleR-np.array([[0,-1,0],[0,0,1],[-1,0,0]]))
    assert max(errors.values())<1e-12,errors

    a,b,g=np.pi/3,np.pi/3,np.pi/4
    first=R(Z,a)
    second=R(first@Y,b)@first
    third=R(second@Z,g)@second
    lower_first=R(Z,g)
    lower_second=R(Y,b)@lower_first
    lower_third=R(Z,a)@lower_second
    frames=[first,second,third,lower_first,lower_second,lower_third]
    assert np.allclose(third,lower_third,atol=1e-13)
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(14,10),facecolor='#fbfcfe')
    titles=[
        r'随体轴 ①：绕 $z$ 转 $\alpha$',
        r'随体轴 ②：绕 $y^\prime$ 转 $\beta$',
        r'随体轴 ③：绕 $z^\prime$ 转 $\gamma$',
        r'固定轴 ①：绕 $z$ 转 $\gamma$',
        r'固定轴 ②：绕 $y$ 转 $\beta$',
        r'固定轴 ③：绕 $z$ 转 $\alpha$']
    colors=['#ba4b38','#32926c','#276da3']
    for k,(frame,title) in enumerate(zip(frames,titles)):
        ax=fig.add_subplot(2,3,k+1,projection='3d')
        ax.set_facecolor('#fbfcfe')
        for j,name in enumerate(['x','y','z']):
            axis=I3[:,j]
            ax.plot([0,axis[0]],[0,axis[1]],[0,axis[2]],color='#a8b1bd',lw=1.5,ls='--')
            ax.text(*(1.08*axis),name,color='#7b8491',fontsize=12)
            vector=frame[:,j]
            ax.quiver(0,0,0,*vector,color=colors[j],linewidth=2.8,arrow_length_ratio=.12)
            ax.text(*(1.08*vector),'$'+name+'_b$',color=colors[j],fontsize=15)
        ax.scatter([0],[0],[0],color='#344054',s=14)
        ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),zlim=(-1.15,1.15))
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=24,azim=36)
        ax.set_axis_off()
        ax.set_title(title,fontsize=17,pad=0,color='#1e293b')
    fig.suptitle('欧拉旋转：两条路线，中间姿态不同，最终姿态相同',fontsize=23,y=.97)
    fig.text(.5,.918,r'$\alpha=\pi/3,\quad\beta=\pi/3,\quad\gamma=\pi/4$'
             '    彩色：刚体轴     灰色虚线：固定实验室轴',ha='center',fontsize=15,color='#485569')
    fig.text(.5,.045,r'$R_{z^\prime}(\gamma)R_{y^\prime}(\beta)R_z(\alpha)'
             r'=R_z(\alpha)R_y(\beta)R_z(\gamma)$'
             '    两边均按最右先作用',ha='center',fontsize=18,color='#1e293b')
    fig.subplots_adjust(left=.015,right=.985,top=.87,bottom=.09,wspace=.02,hspace=.08)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'3.3-欧拉旋转两种路线.{ext}',dpi=170,facecolor=fig.get_facecolor())
    plt.close(fig)
    report={'status':'PASS','random_trials_per_identity':100,'maximum_residuals':errors,
            'rotation_convention':'active column vectors; D=exp(-i phi n.sigma/2)',
            'figure_angles_radians':[a,b,g],
            'figure_final_frame_difference':float(np.max(np.abs(third-lower_third))),
            'worked_example_axis':sample_n.tolist(),'worked_example_angle':'2*pi/3',
            'plot_kind':'analytic teaching illustration; no experimental data'}
    (OUT/'3.3-旋转群与欧拉角核验结果.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
