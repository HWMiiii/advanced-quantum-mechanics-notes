"""Schwinger representation and Wigner d: supplementary, reproducible checks.

Uses hbar=1. Matrix rows are final m', columns are initial m, both descending.
Requires numpy and matplotlib. No textbook files are modified.
"""
from pathlib import Path
from math import factorial, sqrt, sin, cos, pi, comb
import json
import numpy as np

OUT = Path(__file__).resolve().parent
TOL = 5e-12


def fact(x):
    n = int(round(x))
    if n < 0 or abs(n-x) > 1e-10:
        raise ValueError(f'Invalid factorial argument {x}')
    return factorial(n)


def d_element(j, mp, m, beta):
    """Equation (3.426), with exact nonnegative factorial bounds."""
    k_min = int(round(max(0, m-mp)))
    k_max = int(round(min(j+m, j-mp)))
    c, s = cos(beta/2), sin(beta/2)
    norm = sqrt(fact(j+m)*fact(j-m)*fact(j+mp)*fact(j-mp))
    value = 0.0
    for k in range(k_min, k_max+1):
        sign = (-1)**int(round(k-m+mp))
        den = fact(j+m-k)*fact(k)*fact(j-k-mp)*fact(k-m+mp)
        ec = int(round(2*j-2*k+m-mp))
        es = int(round(2*k-m+mp))
        value += sign*norm/den*c**ec*s**es
    return value


def d_matrix(j, beta):
    ms = np.arange(j, -j-0.25, -1.0)
    return np.array([[d_element(j, mp, m, beta) for m in ms] for mp in ms])


def schwinger_matrices(n):
    """Construct number-conserving bilinears directly in the exact N=n sector.

    This avoids testing truncated boson commutators, which fail at the cutoff.
    """
    plus = np.zeros((n+1, n+1), complex)
    ms = np.arange(n/2, -n/2-0.25, -1.0)
    for col in range(n+1):
        np_, nm_ = n-col, col
        if nm_:
            plus[col-1, col] = sqrt(nm_*(np_+1))
    minus = plus.conj().T
    z = np.diag(ms)
    x = (plus+minus)/2
    y = (plus-minus)/(2j)
    return ms, plus, minus, x, y, z


def exponential_from_eigh(generator, beta):
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    return (eigenvectors*np.exp(-1j*beta*eigenvalues))@eigenvectors.conj().T


def run_checks():
    maxima = {}
    counts = {}

    def check(name, difference):
        error = float(np.max(np.abs(difference)))
        maxima[name] = max(maxima.get(name, 0), error)
        counts[name] = counts.get(name, 0)+1
        if error > TOL:
            raise AssertionError(f'{name}: error {error} > {TOL}')

    betas = [-2*pi, -1.13, 0.0, 0.17, pi/2, 2.19, pi, 2*pi, 4*pi]
    compositions = [(0.2,0.3), (-0.7,1.1), (pi/2,pi/2), (pi,pi), (-pi,0.2), (2*pi,0.4)]
    for n in range(9):
        j = n/2
        ms, jp, jm, jx, jy, jz = schwinger_matrices(n)
        eye = np.eye(n+1)
        check('commutator_Jz_Jplus', jz@jp-jp@jz-jp)
        check('commutator_Jplus_Jminus', jp@jm-jm@jp-2*jz)
        check('Casimir', jx@jx+jy@jy+jz@jz-j*(j+1)*eye)
        for beta in betas:
            d = d_matrix(j, beta)
            check('factorial_vs_matrix_exponential', d-exponential_from_eigh(jy,beta))
            check('orthogonality', d.T@d-eye)
            check('inverse', d_matrix(j,-beta)-d.T)
            p = d[:,0]**2
            expected = np.array([comb(n,k)*cos(beta/2)**(2*(n-k))*sin(beta/2)**(2*k) for k in range(n+1)])
            check('highest_weight_binomial', p-expected)
            check('probability_normalization', p.sum()-1)
            mean = np.dot(ms,p)
            var = np.dot(ms**2,p)-mean**2
            check('highest_weight_mean', mean-j*cos(beta))
            check('highest_weight_variance', var-j/2*sin(beta)**2)
        for a,b in compositions:
            check('same_axis_composition', d_matrix(j,a)@d_matrix(j,b)-d_matrix(j,a+b))
        dpi = np.zeros((n+1,n+1))
        for col,m in enumerate(ms):
            dpi[n-col,col] = (-1)**int(round(j-m))
        check('pi_rotation', d_matrix(j,pi)-dpi)
        check('two_pi_rotation', d_matrix(j,2*pi)-(-1)**n*eye)
        check('four_pi_rotation', d_matrix(j,4*pi)-eye)
    result = {
        'status': 'PASS', 'hbar': 1, 'tolerance': TOL,
        'basis': 'Rows m_prime, columns m, both descending from j to -j',
        'j_values': [n/2 for n in range(9)], 'beta_values_radian': betas,
        'composition_pairs_radian': compositions,
        'check_counts': counts, 'maximum_absolute_errors': maxima,
        'total_checks': sum(counts.values()),
        'j_3_over_2_beta_pi_over_2': {
            'd_matrix': d_matrix(1.5,pi/2).tolist(),
            'highest_weight_probabilities': (d_matrix(1.5,pi/2)[:,0]**2).tolist(),
            'mean_m': 0, 'variance_m': 0.75
        },
        'scope': 'Finite numerical examples support, but do not replace, the algebraic proof; factorial sums are not recommended for large j due to cancellation.'
    }
    (OUT/'3.9-施温格模型与旋转矩阵核验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def make_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        matplotlib.rcParams['font.family'] = FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['svg.fonttype'] = 'none'
    fig, (ax, bx) = plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    fig.suptitle('施温格振子模型：固定总数与有限旋转（教学补充）',fontsize=16)
    for np_ in range(5):
        for nm_ in range(5):
            ax.scatter(np_,nm_,s=25,color='#c5cbd2',zorder=1)
    ax.plot([0,3],[3,0],color='#6840a3',lw=2,alpha=.7)
    ax.scatter([0,1,2,3],[3,2,1,0],s=85,color='#6840a3',zorder=3)
    def arrow(a,b,color):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':color,'lw':2.5,'mutation_scale':15},zorder=4)
    arrow((1.08,1.92),(1.92,1.08),'#1b6ca8')
    arrow((1.89,.91),(.91,1.89),'#1b6ca8')
    ax.text(1.68,1.54,r'$J_+$',color='#1b6ca8',fontsize=13)
    ax.text(.96,1.15,r'$J_-$',color='#1b6ca8',fontsize=13)
    arrow((2.05,2.08),(2.94,2.97),'#d15b24')
    arrow((3.12,2.96),(2.23,2.07),'#d15b24')
    ax.text(2.13,2.65,r'$K_+$',color='#d15b24',fontsize=13)
    ax.text(3.08,2.4,r'$K_-$',color='#d15b24',fontsize=13)
    ax.text(2.52,3.44,'紫线：$N=3$，$j=3/2$\n蓝箭头：$j$ 不变，$m$ 改变\n橙箭头：$m$ 不变，$j$ 改变',fontsize=10.5,ha='center',va='center',bbox={'facecolor':'white','edgecolor':'#cccccc','alpha':.95})
    ax.set(xlabel=r'$n_+$',ylabel=r'$n_-$',xlim=(-.35,4.35),ylim=(-.35,4.35),xticks=range(5),yticks=range(5))
    ax.set_aspect('equal')
    ax.set_title('A  两模式占据数格点',fontsize=13)
    ax.grid(alpha=.15)
    beta=np.linspace(0,pi,401)
    for k,color in enumerate(['#6840a3','#1b6ca8','#26906f','#d15b24']):
        p=comb(3,k)*np.cos(beta/2)**(2*(3-k))*np.sin(beta/2)**(2*k)
        label=[r"$m'=3/2$",r"$m'=1/2$",r"$m'=-1/2$",r"$m'=-3/2$"][k]
        bx.plot(beta/pi,p,color=color,label=label,lw=2.3)
        bx.scatter([.5],[comb(3,k)/8],color=color,s=28,zorder=4)
    bx.axvline(.5,color='#888888',ls=':',lw=1.1)
    bx.text(.5,.65,r'$\beta=\pi/2$'+'\n'+r'$P=(1,3,3,1)/8$',fontsize=12,ha='center',bbox={'facecolor':'white','edgecolor':'#cccccc'})
    bx.set(xlabel=r'旋转角 $\beta/\pi$',ylabel=r"$P(m'\mid j=3/2,m=3/2)$",xlim=(0,1),ylim=(0,1.03),xticks=[0,.25,.5,.75,1])
    bx.set_title('B  初态为最高权态，绕 $y$ 轴旋转',fontsize=13)
    bx.legend(loc='upper center',fontsize=10,framealpha=.95,ncol=2)
    bx.grid(alpha=.2)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'3.9-占据数格点与旋转概率.{ext}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    result=run_checks()
    make_figure()
    print(json.dumps({'status':result['status'],'total_checks':result['total_checks'],'max_error':max(result['maximum_absolute_errors'].values())},ensure_ascii=False))
