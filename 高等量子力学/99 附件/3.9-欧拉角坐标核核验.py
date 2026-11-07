from pathlib import Path
import json
import numpy as np

out = Path(__file__).parent
jmax = 3
angles = np.arange(16) * (2*np.pi/16)
x, wx = np.polynomial.legendre.leggauss(12)
alpha, beta, gamma = np.meshgrid(angles, np.arccos(x), angles, indexing='ij')
weights = np.broadcast_to(wx[None,:,None], alpha.shape).ravel()*(2*np.pi/16)**2
a,b,g = alpha.ravel(),beta.ravel(),gamma.ravel()
columns=[]
labels=[]
for j in range(jmax+1):
    ms=np.arange(j,-j-1,-1)
    jp=np.zeros((2*j+1,2*j+1))
    for col,m in enumerate(ms):
        if col>0: jp[col-1,col]=np.sqrt((j-m)*(j+m+1))
    jy=(jp-jp.T)/(2j)
    ev, u=np.linalg.eigh(jy)
    d=np.einsum('ik,pk,jk->pij',u,np.exp(-1j*b[:,None]*ev),u.conj())
    for row,m in enumerate(ms):
        for col,n in enumerate(ms):
            columns.append(np.sqrt((2*j+1)/(8*np.pi**2))*np.exp(-1j*m*a)*d[:,row,col]*np.exp(-1j*n*g))
            labels.append((j,int(m),int(n)))
phi=np.asarray(columns).T
gram=phi.conj().T@(weights[:,None]*phi)
err=float(np.max(np.abs(gram-np.eye(len(labels)))))
coeff=np.zeros(len(labels),dtype=complex)
for lab,c in [((1,1,0),1),((2,-2,1),0.3+0.2j),((3,0,2),-0.7j)]:
    coeff[labels.index(lab)]=c
field=phi@coeff
projections=phi.conj().T@(weights*field)
m2=np.array([m*m for j,m,n in labels])
result=phi@(m2*projections)
expected=phi@(m2*coeff)
err_kernel=float(np.max(np.abs(result-expected)))
assert err<1e-12 and err_kernel<1e-12
report={'status':'PASS','hbar_units':1,'group':'SO(3)','j_max':jmax,'basis_functions':len(labels),'euler_quadrature_points':len(a),'measure_volume':float(sum(weights)),'orthonormality_max_error':err,'kernel_action_max_error':err_kernel,'test_coefficients':[{'j':lab[0],'m':lab[1],'n':lab[2],'coefficient':[float(c.real),float(c.imag)]} for lab,c in zip(labels,coeff) if c!=0],'notes':'Independent spectral Jy calculation; normalized Phi=sqrt((2j+1)/(8pi^2))*D; J3 squared uses first label m.'}
(out/'3.9-欧拉角坐标核核验结果.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))

