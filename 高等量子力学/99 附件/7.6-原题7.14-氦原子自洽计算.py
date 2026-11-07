"""原题7.14。运行 python 本文件.py；加 --figure 图名.png 可重绘图7.7。
NumPy 用于数值积分和 3×3 本征值问题。绘图另需 Pillow。
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ZEFF = 27 / 16
Z = 2
HA_EV = 27.211386245981
PI = np.pi
BOOK = np.array([
    [-6.90876, 2.9882, 2.17546, -1.13109, -78.2649],
    [-6.48303, 2.6350, 1.99548, -1.04472, -78.8379],
    [-6.56468, 2.6978, 2.03054, -1.06127, -78.8478],
])


def basis(r):
    """R_n0(r)，径向积分的测度为 r²dr；径向函数在原点取正。"""
    rows = []
    for n in (1, 2, 3):
        x = 2 * ZEFF * r / n
        laguerre = (1, 2 - x, 3 - 3*x + x*x/2)[n - 1]
        rows.append((2 * ZEFF/n)**1.5 / (np.sqrt(2)*n)
                    * np.exp(-x/2) * laguerre)
    return np.array(rows)


def cumulative_trapezoid(y, step):
    return np.r_[0., np.cumsum((y[1:] + y[:-1]) * (step/2))]


def hartree(r, density, step):
    enclosed = cumulative_trapezoid(density*r*r, step)
    tail = cumulative_trapezoid((density*r)[::-1], step)[::-1]
    # V(0)=4π∫n(r')r'dr'。右端不做除零运算。
    potential = 4*PI*tail
    potential[1:] += 4*PI*enclosed[1:] / r[1:]
    return potential


def xc(density, rs_has_pi, correlation_sign):
    """返回每粒子能 ε_xc，及势 d(n ε_xc)/dn；n=0 时取连续极限。"""
    positive = density > 0
    d = density[positive]
    rs = ((4/3)*(PI if rs_has_pi else 1)*d)**(-1/3)
    root = np.sqrt(rs)
    denom = 1 + .1018*root + .102582*rs
    ex = -1.174 * 3/(4*PI) * np.cbrt(3*PI**2*d)
    ec = correlation_sign * .0233504 / denom
    vc = ec * (1 + (.1018*root/2 + .102582*rs)/(3*denom))
    energy = np.zeros_like(density)
    potential = np.zeros_like(density)
    energy[positive] = ex + ec
    potential[positive] = 4*ex/3 + vc
    return energy, potential


@dataclass
class Step:
    number: int
    coefficients: np.ndarray
    components: np.ndarray
    residual: float
    density: np.ndarray


class RadialBasis:
    def __init__(self, step=.0025, radius=60.):
        count = 2 * round(radius / (2 * step))
        self.r = np.linspace(0, radius, count+1)
        self.h = self.r[1]
        self.w = np.full(count+1, 2*self.h/3)
        self.w[1::2] = 4*self.h/3
        self.w[[0, -1]] = self.h/3
        self.R = basis(self.r)
        self.vol = 4*PI*self.w*self.r**2
        overlap = (self.R * (self.w*self.r**2)) @ self.R.T
        if np.max(np.abs(overlap-np.eye(3))) > 1.e-8:
            raise ValueError('基函数在本网格没有正交归一，请扩大径向范围或加密网格。')
        inverse_radius = (self.R * (self.w*self.r)) @ self.R.T
        self.T = np.diag(-ZEFF**2/(2*np.arange(1,4)**2)) + ZEFF*inverse_radius
        self.Vext = -Z*inverse_radius

    def run(self, start, has_pi=True, corr_sign=-1, steps=60, tol=1e-8):
        q = ZEFF if start == 'diffuse' else 2*ZEFF
        density = q**3/(4*PI)*np.exp(-q*self.r)
        initial = density.copy()
        history = []
        for i in range(1, steps+1):
            vh = hartree(self.r, density, self.h)
            _, vxc = xc(density, has_pi, corr_sign)
            interaction = (self.R * (self.w*self.r**2*(vh + vxc))) @ self.R.T
            _, vectors = np.linalg.eigh(self.T + self.Vext + interaction)
            c = vectors[:, 0]
            if c[0] < 0:
                c = -c
            orbital = c @ self.R / np.sqrt(4*PI)
            new = 2*orbital**2
            if abs(self.vol @ new - 2) > 1e-8:
                raise ArithmeticError('电子数不守恒。')
            new_vh = hartree(self.r, new, self.h)
            eps, _ = xc(new, has_pi, corr_sign)
            components = [2*c@self.Vext@c, 2*c@self.T@c,
                          (self.vol*new)@new_vh/2, (self.vol*new)@eps]
            energy = sum(components) * HA_EV
            residual = self.vol @ np.abs(new-density)
            history.append(Step(i, c, np.array([*components, energy]),
                                residual, new))
            density = new
            if i >= 3 and residual < tol:
                return initial, history
        raise ArithmeticError(f'{steps}步内未达到密度收敛要求，末步残差 {residual:g}')


def report(name, history):
    print('\n' + name)
    print('k     U_ext       T_KS        U_ee       U_xc       E/eV        Dn')
    for row in [*history[:3], history[-1]]:
        values = ' '.join(f'{v:11.6f}' for v in row.components)
        print(f'{row.number:2d} {values} {row.residual:.3g}')
    print('first c =', history[0].coefficients)


def draw(path, grid, initial, history):
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1250, 750
    canvas = Image.new('RGB', (width,height), 'white')
    draw = ImageDraw.Draw(canvas)
    for face in ('msyh.ttc', 'NotoSansCJK-Regular.ttc', 'DejaVuSans.ttf', 'arial.ttf'):
        try:
            ImageFont.truetype(face, 18)
            break
        except OSError:
            continue
    else:
        face = None
    chinese = face in ('msyh.ttc', 'NotoSansCJK-Regular.ttc')
    def font(size):
        return ImageFont.truetype(face, size) if face else ImageFont.load_default()
    title = '氦原子电子密度的前三次迭代' if chinese else 'Helium radial electron density'
    subtitle = ('独立重算图7.7   ·   三个氢样 s 轨道' if chinese else
                'Independent calculation of Fig. 7.7   |   1s, 2s, 3s basis')
    draw.text((70,30), title, fill='#132d3a', font=font(43))
    draw.text((70,92), subtitle,
              fill='#526675', font=font(26))
    left,right,top,bottom = 100,940,183,640
    def xy(r,v):
        return (left+(right-left)*r/5, bottom-(bottom-top)*v/.16)
    for y in (0,.05,.10,.15):
        _, pix = xy(0,y)
        draw.line((left,pix,right,pix),fill='#dfe5e8',width=2)
        draw.text((left-20,pix),f'{y:.2f}',font=font(22),fill='#263e4c',anchor='rm')
    for x in range(6):
        pix,_ = xy(x,0)
        draw.line((pix,bottom,pix,bottom+6),fill='#263e4c',width=2)
        draw.text((pix,bottom+12),str(x),font=font(23),fill='#263e4c',anchor='mt')
    draw.line((left,top,left,bottom,right,bottom),fill='#263e4c',width=3)
    draw.text((left,top-40),'r² n(r)   [1/a0]',font=font(23),fill='#263e4c')
    draw.text(((left+right)/2,700),'r / a0',font=font(24),fill='#263e4c',anchor='mm')
    styles = [('Initial','#616e78',None),('Iteration 1','#c06f16',(15,8)),
              ('Iteration 2','#0068aa',(3,5)),('Iteration 3','#8d3975',None)]
    for k,(density,(label,color,pattern)) in enumerate(zip(
            [initial, *(item.density for item in history[:3])], styles)):
        if chinese:
            label = ('初始密度','第一次','第二次','第三次')[k]
        radius = np.linspace(0,5,1300)
        values = radius**2*np.interp(radius,grid.r,density)
        points = [xy(r,v) for r,v in zip(radius,values)]
        # 分段虚线与点线可在黑白打印时区分迭代。
        if pattern is None:
            draw.line(points,fill=color,width=4)
        else:
            for j in range(0,len(points)-1,sum(pattern)):
                draw.line(points[j:j+pattern[0]+1],fill=color,width=4)
        h = top+45+65*k
        if pattern is None:
            draw.line((963,h,1016,h),fill=color,width=4)
        else:
            for j in range(963,1016,sum(pattern)):
                draw.line((j,h,min(1016,j+pattern[0]),h),fill=color,width=4)
        draw.text((1027,h),label,font=font(25),fill=color,anchor='lm')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',type=Path)
    args = parser.parse_args()
    grid = RadialBasis()
    initial, rows = grid.run('diffuse')
    report('Sun correlation / diffuse first density', rows)
    for k,density in enumerate([initial, *(row.density for row in rows[:3])]):
        peak = np.argmax(grid.r**2*density)
        print(f'k={k}, peak at r={grid.r[peak]:.4f}, '
              f'r²n={grid.r[peak]**2*density[peak]:.6f}, electrons={grid.vol@density:.9f}')
    print('independent minus printed (Ha / final column eV):')
    print(np.array([row.components for row in rows[:3]]) - BOOK)
    for name, has_pi, sign in [('literal Chinese print',False,1),
                               ('single minus, omit pi',False,-1)]:
        _, other = grid.run('diffuse', has_pi, sign)
        report(name, other)
    _, better = grid.run('hydrogenic')
    report('hydrogenic first density', better)
    for h, rmax in ((.005,60),(.00125,60),(.0025,80)):
        _, test = RadialBasis(h,rmax).run('diffuse')
        print(f'check h={h:g}, R={rmax:g} : E1={test[0].components[-1]:.8f}, '
              f'E3={test[2].components[-1]:.8f}, Econv={test[-1].components[-1]:.8f} eV')
    if args.figure:
        draw(args.figure, grid, initial, rows)


if __name__ == '__main__':
    main()
