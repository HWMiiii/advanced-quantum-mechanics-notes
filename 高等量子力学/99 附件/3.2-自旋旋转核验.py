"""Reproduce section 3.2's dimensionless teaching plot and matrix checks.

Requires numpy and matplotlib. No experimental data are used.
Angles are radians; active rotation D=exp(-i phi n.sigma/2).
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path(__file__).resolve().parent
I=np.eye(2,dtype=complex)
SIG=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],dtype=complex)

def nsigma(n):return np.einsum('i,ijk->jk',n,SIG)
def rotation(n,phi):return np.cos(phi/2)*I-1j*np.sin(phi/2)*nsigma(n)
def bloch(chi):return np.real(np.einsum('j,ijk,k->i',chi.conj(),SIG,chi))

def main():
    rng=np.random.default_rng(32026)
    errs={k:0. for k in ['pauli_product','unitarity','determinant','bloch_rotation','interference_probability','cayley','euler','direction_eigenstate']}
    for _ in range(50):
        n=rng.normal(size=3);n/=np.linalg.norm(n)
        a,b=rng.normal(size=(2,3));phi=rng.uniform(-4*np.pi,4*np.pi)
        chi=rng.normal(size=2)+1j*rng.normal(size=2);chi/=np.linalg.norm(chi)
        D=rotation(n,phi);r=bloch(chi)
        expected=r*np.cos(phi)+np.cross(n,r)*np.sin(phi)+n*np.dot(n,r)*(1-np.cos(phi))
        delta=rng.uniform(-np.pi,np.pi)
        amp=(chi+np.exp(1j*delta)*D@chi)/2
        p=.5*(1+np.cos(phi/2)*np.cos(delta)+np.dot(n,r)*np.sin(phi/2)*np.sin(delta))
        a0=rng.normal();Q=a0*a0+np.dot(a,a)
        direct=(a0*I+1j*nsigma(a))@np.linalg.inv(a0*I-1j*nsigma(a))
        closed=((a0*a0-np.dot(a,a))*I+2j*a0*nsigma(a))/Q
        alpha,beta,gamma=rng.uniform(-5,5,size=3)
        Euler=rotation([0,0,1],alpha)@rotation([0,1,0],beta)@rotation([0,0,1],gamma)
        u=(alpha+gamma)/2;v=(alpha-gamma)/2
        c=np.cos(beta/2)*np.cos(u)
        w=np.array([-np.sin(beta/2)*np.sin(v),np.sin(beta/2)*np.cos(v),np.cos(beta/2)*np.sin(u)])
        theta=rng.uniform(0,np.pi);varphi=rng.uniform(-np.pi,np.pi)
        nv=np.array([np.sin(theta)*np.cos(varphi),np.sin(theta)*np.sin(varphi),np.cos(theta)])
        chiv=np.array([np.cos(theta/2),np.exp(1j*varphi)*np.sin(theta/2)])
        residuals={
            'pauli_product':nsigma(a)@nsigma(b)-np.dot(a,b)*I-1j*nsigma(np.cross(a,b)),
            'unitarity':D.conj().T@D-I,'determinant':np.linalg.det(D)-1,
            'bloch_rotation':bloch(D@chi)-expected,
            'interference_probability':np.vdot(amp,amp).real-p,
            'cayley':direct-closed,'euler':Euler-(c*I-1j*nsigma(w)),
            'direction_eigenstate':nsigma(nv)@chiv-chiv}
        for name,value in residuals.items():errs[name]=max(errs[name],float(np.max(np.abs(value))))
    assert max(errs.values())<1e-12,errs
    D=rotation(np.array([1.,2,2])/3,np.pi/3)
    ch=D@np.array([1.,0]);probs=np.abs(ch)**2
    assert np.allclose(probs,[31/36,5/36])
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.size']=12
    x=np.linspace(0,4,1201);phi=np.pi*x
    blue='#266897';red='#b64c36';gray='#707b88'
    fig,axes=plt.subplots(3,1,figsize=(10.5,9.4),sharex=True,facecolor='#fafbfd')
    axes[0].plot(x,np.cos(phi),c=blue,lw=2.2,label=r'$r_x=\cos\phi$')
    axes[0].plot(x,np.sin(phi),c=red,lw=1.8,ls='--',label=r'$r_y=\sin\phi$')
    axes[0].set_title('平均方向：转过 2π 后复原（初态沿 +x，绕 +z 转）',loc='left',fontsize=13)
    axes[0].set_ylabel('Bloch 向量分量')
    axes[1].plot(x,np.cos(phi/2),c=blue,lw=2.2,label=r'$\langle +x|D_z(\phi)|+x\rangle=\cos(\phi/2)$')
    axes[1].scatter([0,2,4],[1,-1,1],c=[blue,red,blue],s=45,zorder=3)
    axes[1].annotate('负号',xy=(2,-1),xytext=(2.2,-.65),color=red,arrowprops={'arrowstyle':'->','color':red})
    axes[1].set_title('旋量与初态的重叠：2π 时为 −1，4π 时为 +1',loc='left',fontsize=13)
    axes[1].set_ylabel('实重叠振幅')
    axes[2].plot(x,(1+np.cos(phi/2))/2,c=blue,lw=2.2,label=r'$\delta=0:\ P=(1+\cos(\phi/2))/2$')
    axes[2].plot(x,np.full_like(x,.5),c=gray,lw=1.6,ls='--',label=r'未极化且 $\delta=\pi/2:\ P=1/2$')
    axes[2].scatter([0,2,4],[1,0,1],c=[blue,red,blue],s=45,zorder=3)
    axes[2].set_title('相干两臂输出：只有一臂旋转；相位工作点决定条纹',loc='left',fontsize=13)
    axes[2].set_ylabel('输出端概率');axes[2].set_xlabel(r'旋转角 $\phi/\pi$')
    for k,ax in enumerate(axes):
        ax.set_facecolor('#fafbfd');ax.set_xlim(0,4)
        ax.set_ylim((-.13,1.18) if k==2 else (-1.22,1.3))
        ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=10,frameon=False)
        for v in [0,2,4]:ax.axvline(v,color=gray,lw=.8,alpha=.4)
    axes[2].set_xticks(np.arange(0,4.1,.5))
    fig.suptitle('§3.2  三种不同的“复现”',fontsize=18,y=.995)
    fig.tight_layout(h_pad=2.1,rect=(0,0,1,.97))
    target=OUT/'3.2-自旋半角与干涉周期.png'
    fig.savefig(target,dpi=170);fig.savefig(target.with_suffix('.svg'));plt.close(fig)
    result={'status':'PASS','active_rotation_convention':'D=cos(phi/2)I-i sin(phi/2)n.sigma',
            'random_trials_per_identity':50,'maximum_residuals':errs,
            'example_axis':[1/3,2/3,2/3],'example_angle':'pi/3','example_probabilities':probs.tolist(),
            'interferometer':'ideal balanced coherent arms; B arm rotated; P at one output',
            'unpolarized_delta_pi_over_2':'constant probability 1/2; no field fringe at this phase bias'}
    (OUT/'3.2-自旋旋转核验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
