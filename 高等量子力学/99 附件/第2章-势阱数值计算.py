# -*- coding: utf-8 -*-
"""教材第二章 2.32、2.33、2.34 的数值计算及归一化波函数图。

依赖：Python 3、numpy>=2、Pillow；默认使用 Windows 微软雅黑字体。
安装：python -m pip install "numpy>=2" Pillow
运行：python "第2章-势阱数值计算.py"
输出：本程序所在文件夹下的“第2章-势阱计算结果”，含三张图与数值核验结果.json。
程序同时用有限差分法独立检查能级。中央势垒求根分支按原题
b/a=1/3、V0/E*=20 或 10 设置，不是任意参数的通用势阱求解器。
"""

from pathlib import Path
import math
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / '第2章-势阱计算结果'
FONT_PATH = r'C:\Windows\Fonts\msyh.ttc'


def bisect(fun, lo, hi, tol=2e-14):
    fl, fh = fun(lo), fun(hi)
    if fl == 0:
        return lo
    if fh == 0:
        return hi
    if fl * fh >= 0:
        raise ValueError((lo, hi, fl, fh))
    for _ in range(100):
        mid = (lo + hi) / 2
        fm = fun(mid)
        if fm == 0 or hi - lo < tol:
            return mid
        if fl * fm < 0:
            hi, fh = mid, fm
        else:
            lo, fl = mid, fm
    return (lo + hi) / 2


def finite_well(eta=10.0):
    ans = []
    n = 0
    top = math.sqrt(eta)
    while n * math.pi / 2 < top:
        q = bisect(lambda q: q + math.asin(min(q / top, 1.0)) - (n + 1) * math.pi / 2, 0.0, top)
        r = math.sqrt(eta - q*q)
        norm = 1 / math.sqrt(1 + 1/r)
        parity = 1 if n % 2 == 0 else -1
        def wf(y, q=q, r=r, norm=norm, parity=parity):
            y = np.asarray(y)
            if parity == 1:
                return norm * np.where(np.abs(y) <= 1, np.cos(q*y), math.cos(q)*np.exp(-r*(np.abs(y)-1)))
            return norm * np.where(np.abs(y) <= 1, np.sin(q*y), np.sign(y)*math.sin(q)*np.exp(-r*(np.abs(y)-1)))
        yin = np.linspace(-1, 1, 20001)
        ytail = np.linspace(1, 1+12/r, 20001)
        integral = float(np.trapezoid(wf(yin)**2, yin) + 2*np.trapezoid(wf(ytail)**2, ytail))
        m = np.array([
            [math.exp(-r), -np.exp(-1j*q), -np.exp(1j*q), 0],
            [r*math.exp(-r), -1j*q*np.exp(-1j*q), 1j*q*np.exp(1j*q), 0],
            [0, np.exp(1j*q), np.exp(-1j*q), -math.exp(-r)],
            [0, 1j*q*np.exp(1j*q), -1j*q*np.exp(-1j*q), r*math.exp(-r)]
        ], dtype=complex)
        ans.append(dict(n=n, parity='even' if parity == 1 else 'odd', q=q, r=r,
                        energy_unit=q*q, epsilon=q*q/eta, norm=norm,
                        norm_integral=integral, matching_singular_value=float(np.linalg.svd(m, compute_uv=False)[-1]), wf=wf))
        n += 1
    return ans


def central_barrier(v, beta=1/3):
    ell = 1-beta
    result = []
    for parity in (1, -1):
        def f(e):
            q, kap = math.sqrt(e), math.sqrt(max(0, v-e))
            inside = (kap*math.tanh(kap*beta) if parity == 1 else
                      (kap/math.tanh(kap*beta) if kap > 1e-9 else 1/beta))
            return inside + q / math.tan(q*ell)
        # For v=10 or 20 only the first negative-cotangent interval is accessible.
        lo = (math.pi/(2*ell))**2 + 1e-10
        if lo >= v or f(v) >= 0:
            continue
        e = bisect(f, lo, v)
        q, kap = math.sqrt(e), math.sqrt(v-e)
        gb = math.cosh(kap*beta) if parity == 1 else math.sinh(kap*beta)
        outer_integral = ell - math.sin(2*q*ell)/(2*q)
        inner_base = beta + math.sinh(2*kap*beta)/(2*kap) if parity == 1 else math.sinh(2*kap*beta)/(2*kap) - beta
        inner_integral = math.sin(q*ell)**2 / gb**2 * inner_base
        norm = 1/math.sqrt(inner_integral + outer_integral)
        def wf(y, q=q, kap=kap, norm=norm, parity=parity, gb=gb):
            y = np.asarray(y)
            inside = math.sin(q*ell) * (np.cosh(kap*y) if parity == 1 else np.sinh(kap*y)) / gb
            outside = np.sin(q*(1-np.abs(y))) * (1 if parity == 1 else np.sign(y))
            return norm*np.where(np.abs(y) <= beta, inside, outside)
        y = np.linspace(-1,1,60001)
        g = math.cosh(kap*beta) if parity == 1 else math.sinh(kap*beta)
        gp = kap*math.sinh(kap*beta) if parity == 1 else kap*math.cosh(kap*beta)
        mat = np.array([[g,-math.sin(q*ell)],[gp,q*math.cos(q*ell)]])
        result.append(dict(v=v, parity='even' if parity == 1 else 'odd', energy=e, q=q, kappa=kap,
                           norm=norm, central_probability=norm*norm*inner_integral,
                           norm_integral=float(np.trapezoid(wf(y)**2,y)),
                           matching_singular_value=float(np.linalg.svd(mat,compute_uv=False)[-1]), wf=wf))
    return sorted(result, key=lambda s:s['energy'])


def sturm_count(diag, off_squared, energy):
    count = 0
    pivot = diag[0]-energy
    if pivot < 0: count += 1
    for d in diag[1:]:
        if abs(pivot) < 1e-100: pivot = -1e-100
        pivot = d-energy-off_squared/pivot
        if pivot < 0: count += 1
    return count


def fd_levels(y, potential, low, high, wanted=None):
    h = float(y[1]-y[0])
    diag = (2/h**2+potential).tolist()
    off2 = 1/h**4
    count = sturm_count(diag, off2, high)
    levels = []
    for n in range(count if wanted is None else min(count,wanted)):
        lo, hi = low, high
        for _ in range(44):
            mid = (lo+hi)/2
            if sturm_count(diag,off2,mid) <= n: lo=mid
            else: hi=mid
        levels.append((lo+hi)/2)
    return dict(count_below_upper_bound=count, energies=levels, spacing=h, box_edge=float(y[-1]+h))


def delta_kappa(d, parity):
    if parity == 1:
        return bisect(lambda r: 2*r-1-math.exp(-r*d),0.5,1.0)
    if d <= 2:
        return None
    return bisect(lambda r: 2 + math.expm1(-r*d)/r if r else 2-d, 0, 0.5)


def delta_wf(u,d,parity):
    r = delta_kappa(d,parity)
    if r is None: return None
    x = r*d
    den = 1+(1+x)*math.exp(-x) if parity == 1 else -math.expm1(-x)-x*math.exp(-x)
    norm = math.sqrt(r/(2*den))
    u=np.asarray(u)
    return norm*(np.exp(-r*np.abs(u-d/2))+parity*np.exp(-r*np.abs(u+d/2)))


def font(size):
    return ImageFont.truetype(FONT_PATH,size)


class SafeDraw:
    def __init__(self, draw):
        self.draw = draw
    def __getattr__(self, name):
        return getattr(self.draw, name)
    def text(self, xy, text, *args, **kwargs):
        # Microsoft YaHei lacks these two glyphs on this host.
        text = str(text).replace('ℏ', 'hbar').replace('₀', '0')
        return self.draw.text(xy, text, *args, **kwargs)


def canvas(title, subtitle, size=(1900,1300)):
    im=Image.new('RGB',size,'#f5f6f8'); d=SafeDraw(ImageDraw.Draw(im))
    d.text((70,35),title,font=font(42),fill='#162b41')
    d.text((70,100),subtitle,font=font(24),fill='#4e6172')
    return im,d


def panel(draw, box, title, subtitle, xlim, ylim, curves, shades=(), verticals=(), xlabel='y = x/a', ylabel='f = √a ψ'):
    x,y,w,h=box
    draw.rounded_rectangle((x,y,x+w,y+h),radius=18,fill='white',outline='#dce3ea',width=2)
    draw.text((x+24,y+15),title,font=font(27),fill='#1c3347')
    draw.text((x+24,y+55),subtitle,font=font(20),fill='#536575')
    left,top,right,bottom=x+75,y+110,x+w-30,y+h-70
    def px(v): return left+(v-xlim[0])/(xlim[1]-xlim[0])*(right-left)
    def py(v): return bottom-(v-ylim[0])/(ylim[1]-ylim[0])*(bottom-top)
    for a,b in shades:
        draw.rectangle((px(max(a,xlim[0])),top,px(min(b,xlim[1])),bottom),fill='#f4e7d3')
    for xx in np.linspace(xlim[0],xlim[1],5):
        xp=px(xx); draw.line((xp,top,xp,bottom),fill='#e6ebef')
        label=f'{xx:.3g}'
        draw.text((xp-24,bottom+12),label,font=font(19),fill='#526274')
    for yy in np.linspace(ylim[0],ylim[1],5):
        yp=py(yy); draw.line((left,yp,right,yp),fill='#e6ebef')
        draw.text((left-68,yp-12),f'{yy:.2g}',font=font(18),fill='#526274')
    if ylim[0] < 0 < ylim[1]: draw.line((left,py(0),right,py(0)),fill='#8b9bab',width=2)
    for xx in verticals:
        if xlim[0] <= xx <= xlim[1]:
            xp=px(xx)
            for yp in range(int(top),int(bottom),13):draw.line((xp,yp,xp,min(yp+7,bottom)),fill='#ae8052',width=2)
    draw.line((left,top,left,bottom,right,bottom),fill='#586d7c',width=2)
    for xs,ys,color,label in curves:
        pts=[(px(float(xx)),py(float(yy))) for xx,yy in zip(xs,ys) if xlim[0]<=xx<=xlim[1] and ylim[0]<=yy<=ylim[1]]
        if len(pts)>1:draw.line(pts,fill=color,width=4)
    draw.text((left,top-30),ylabel,font=font(18),fill='#385164')
    draw.text((right-135,bottom+40),xlabel,font=font(18),fill='#385164')
    return


def figures(wells, barriers):
    im,d=canvas('2.32  有限方阱 η = 10：全部三个束缚态','横轴 y=x/a；f(y)=√a ψ(x)，每条曲线均满足 ∫|f|²dy=1。上排显示尾部，下排放大阱内。')
    colors=['#156b8a','#b35335','#7357a3']
    for i,s in enumerate(wells):
        L=max(3.,1+6/s['r']); xx=np.linspace(-L,L,7001)
        inside=np.linspace(-1,1,2001); ymax=max(float(np.max(np.abs(s['wf'](xx)))),float(np.max(np.abs(s['wf'](inside)))))*1.15
        parity='偶' if s['parity']=='even' else '奇'
        panel(d,(55+i*620,160,590,490),f'n={i} · {parity}宇称',f'E/V₀={s["epsilon"]:.9f}，αa={s["r"]:.6f}',(-L,L),(-ymax,ymax),[(xx,s['wf'](xx),colors[i],'')],shades=[(-1,1)],verticals=[-1,1])
        xx=np.linspace(-1.3,1.3,3001)
        panel(d,(55+i*620,680,590,490),'阱内及边缘放大',f'归一化系数 N={s["norm"]:.8f}',(-1.3,1.3),(-ymax,ymax),[(xx,s['wf'](xx),colors[i],'')],shades=[(-1,1)],verticals=[-1,1])
    d.text((70,1210),'浅色区域：|y|≤1。第三态能量很接近阱顶，因此有很长的指数尾部；不能只用阱内积分归一化。',font=font(23),fill='#40566a')
    im.save(OUT/'2.5-习题2.32-有限方阱归一化波函数.png')

    states=barriers[20.0]+barriers[10.0]
    im,d=canvas('2.33  硬墙内的中央势垒：低于势垒的归一化态','β=b/a=1/3；ν=V₀/(ℏ²/2ma²)。浅色区为中央势垒，左右端点为硬墙。',size=(1900,850))
    for i,s in enumerate(states):
        xx=np.linspace(-1,1,5001); yy=s['wf'](xx); lim=max(np.max(np.abs(yy))*1.13,1.05)
        par='偶' if s['parity']=='even' else '奇'
        panel(d,(55+i*620,170,590,560),f'ν={s["v"]:g} · {par}宇称',f'E/(ℏ²/2ma²)={s["energy"]:.8f}',(-1,1),(-lim,lim),[(xx,yy,colors[i],'')],shades=[(-1/3,1/3)],verticals=[-1,-1/3,1/3,1])
    d.text((70,775),'ν=20 有两个 E<V₀ 的态；ν=10 只有一个 E<V₀ 的态。两个问题在势垒顶以上仍各有无限多个离散能级。',font=font(23),fill='#40566a')
    im.save(OUT/'2.5-习题2.33-中央势垒归一化波函数.png')

    im,d=canvas('2.34  双 δ 势：强度平分，奇态在 d>2 才束缚','κ₀=mV₀b/ℏ²，d=κ₀a；能量单位 B₀=ℏ²κ₀²/(2m)，坐标 u=κ₀x。')
    ds=np.linspace(0,9,1001); even=np.array([-delta_kappa(v,1)**2 for v in ds]); od=np.array([-delta_kappa(v,-1)**2 if v>2 else np.nan for v in ds])
    panel(d,(60,160,1780,410),'束缚态能量随两中心距离的变化','蓝色：偶态；橙色：奇态。虚线 d=2 是零能阈值，阈值本身不属于正规束缚态。',(0,9),(-1.05,.05),[(ds,even,colors[0],''),(ds[ds>2],od[ds>2],colors[1],'')],verticals=[2],xlabel='d = κ₀a',ylabel='E/B₀')
    for i,dist in enumerate([1.,4.,8.]):
        u=np.linspace(-14,14,6001); curves=[(u,delta_wf(u,dist,1),colors[0],'')]
        if dist>2:curves.append((u,delta_wf(u,dist,-1),colors[1],''))
        par='只有偶束缚态' if dist<=2 else '偶、奇两个束缚态'
        panel(d,(55+i*620,610,590,560),f'd={dist:g}：{par}','f(u)=ψ(x)/√κ₀；每条曲线全空间归一化',(-14,14),(-.8,.8),curves,verticals=[-dist/2,dist/2],xlabel='u = κ₀x',ylabel='f(u)')
    d.text((70,1220),'下排的竖虚线标记两个 δ 中心。图示距离为代表性参数；一般 a 的本征函数及归一化常数见笔记。',font=font(23),fill='#40566a')
    im.save(OUT/'2.5-习题2.34-双delta能级与归一化波函数.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wells=finite_well(); barriers={v:central_barrier(v) for v in [20.,10.]}
    L=math.ceil((1+10/wells[-1]['r'])*50)/50
    y=np.arange(-L+.02,L-.01,.02)
    pot=np.where(np.abs(y)<1-1e-8,0.,10.)
    pot[np.isclose(np.abs(y),1,atol=1e-8)]=5.
    fdwell=fd_levels(y,pot,0.,10.)
    fdbar={}
    for v in [20.,10.]:
        y=np.linspace(-1,1,1201)[1:-1]
        pot=np.where(np.abs(y)<1/3-1e-10,v,0.)
        pot[np.isclose(np.abs(y),1/3,atol=1e-10)]=v/2
        fdbar[v]=fd_levels(y,pot,0.,v)
    y=np.linspace(-30,30,6001)[1:-1]; pot=np.zeros_like(y)
    for c in [-2,2]:pot[np.argmin(abs(y-c))]=-100.
    fddelta=fd_levels(y,pot,-2.,0.)
    delta=[]
    for dist in [1.,4.,8.]:
        for par in [1,-1]:
            r=delta_kappa(dist,par)
            if r is None:continue
            u=np.linspace(-dist/2-14/r,dist/2+14/r,100001)
            delta.append(dict(d=dist,parity='even' if par==1 else 'odd',r=r,energy_over_B0=-r*r,norm_integral=float(np.trapezoid(delta_wf(u,dist,par)**2,u))))
    figures(wells,barriers)
    clean=lambda ss:[{k:v for k,v in s.items() if k!='wf'} for s in ss]
    report=dict(finite_well=clean(wells),finite_well_fd=fdwell,central_barrier={str(k):clean(v) for k,v in barriers.items()},central_barrier_fd=fdbar,double_delta=delta,double_delta_d4_fd=fddelta)
    report_text=json.dumps(report,ensure_ascii=False,indent=2)
    (OUT/'数值核验结果.json').write_text(report_text,encoding='utf-8')
    print(report_text)


if __name__=='__main__':
    main()
