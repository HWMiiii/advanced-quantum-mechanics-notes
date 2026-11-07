"""樱井《现代量子力学》原题6.13：写出图6.15和相应的有效势图。仅用Python标准库。"""
import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape

ELL = 3
DEPTH = 5.5 ** 2  # 正的阱深，单位 E_R = hbar^2/(2mR^2)
STEM = '6.7-原题6.13-'
BLUE, ORANGE, GREEN = '#165987', '#b44a1e', '#08796b'
INK, GRID = '#1a3440', '#d9e3e8'


def spherical_wave(z, kind):
    """给出 j_l 或 n_l 的值及 z 乘上其导数。用于本题 x >= 1.31 的窗口。"""
    if kind == 'j':
        previous = math.sin(z) / z
        current = (previous - math.cos(z)) / z
    else:
        previous = -math.cos(z) / z
        current = (previous - math.sin(z)) / z
    for degree in range(1, ELL):
        previous, current = current, (2 * degree + 1) * current / z - previous
    return current, z * previous - (ELL + 1) * current


def matching(x):
    """外解为 j_l cos(delta) - n_l sin(delta)，返回 tan(delta) 的分子、分母。"""
    y = math.sqrt(x * x + DEPTH)
    inside, inside_derivative = spherical_wave(y, 'j')
    j, dj = spherical_wave(x, 'j')
    n, dn = spherical_wave(x, 'n')
    return inside * dj - inside_derivative * j, inside * dn - inside_derivative * n


def scattering(x):
    numerator, denominator = matching(x)
    phase = math.atan2(numerator, denominator) % math.pi  # 图中连续的 0 到 pi 分支
    limit = 4 * math.pi * (2 * ELL + 1) / x ** 2
    cross_section = limit * numerator ** 2 / (denominator ** 2 + numerator ** 2)
    return phase, cross_section, limit


def bisect(function, low, high):
    sign = function(low)
    if sign * function(high) >= 0:
        raise ValueError('区间两端未夹住该根')
    for _ in range(64):
        mid = (low + high) / 2
        value = function(mid)
        if value == 0:
            return mid
        if sign * value > 0:
            low, sign = mid, value
        else:
            high = mid
    return (low + high) / 2


def derivative(function, x, h=1e-5):
    return (function(x - 2*h) - 8*function(x - h)
            + 8*function(x + h) - function(x + 2*h)) / (12*h)


def resonance():
    xr = bisect(lambda x: matching(x)[1], 1.39, 1.43)
    def cotangent(x):
        numerator, denominator = matching(x)
        return denominator / numerator
    width = -4*xr / derivative(cotangent, xr)  # d/d(E/E_R) = (2x)^(-1) d/dx
    cross = lambda x: scattering(x)[1]
    peak = bisect(lambda x: derivative(cross, x), 1.40, 1.42)
    half = lambda x: cross(x) - cross(peak) / 2
    left = bisect(half, 1.36, peak)
    right = bisect(half, peak, 1.47)
    return xr, peak, width, left, right


def integrate_inside(x, steps):
    """独立计算正则约化径向解：Frobenius级数起步，RK4积分到 rho = 1。"""
    rho = 0.08
    y2 = DEPTH + x*x
    term = 1.0
    u = rho ** (ELL + 1)
    du = (ELL + 1) * rho ** ELL
    for order in range(1, 8):
        term *= -y2 * rho ** 2 / (2 * order * (2*ELL + 2*order + 1))
        u += rho ** (ELL + 1) * term
        du += (ELL + 1 + 2*order) * rho ** ELL * term
    step = (1 - rho) / steps

    def acceleration(r, value):
        return (ELL * (ELL + 1) / r ** 2 - y2) * value

    for _ in range(steps):
        a1, b1 = du, acceleration(rho, u)
        a2 = du + step*b1/2
        b2 = acceleration(rho + step/2, u + step*a1/2)
        a3 = du + step*b2/2
        b3 = acceleration(rho + step/2, u + step*a2/2)
        a4 = du + step*b3
        b4 = acceleration(rho + step, u + step*a3)
        u += step*(a1 + 2*a2 + 2*a3 + a4)/6
        du += step*(b1 + 2*b2 + 2*b3 + b4)/6
        rho += step
    return u, du


def numerical_s_matrix(x, steps):
    u, du = integrate_inside(x, steps)
    j, dj = spherical_wave(x, 'j')
    n, dn = spherical_wave(x, 'n')
    # 约化外出射波 rho*(j+i*n) 在边界的值、导数。
    outgoing = complex(j, n)
    outgoing_derivative = outgoing + complex(dj, dn)
    determinant = u*outgoing_derivative - du*outgoing
    return -determinant.conjugate() / determinant


def check_results(xr):
    errors = []
    for steps in (512, 1024):
        points = [1.32, 1.35, 1.39, 1.40, xr, 1.42, 1.45, 1.50]
        differences = []
        for x in points:
            num, den = matching(x)
            exact = complex(den, num) / complex(den, -num)
            differences.append(abs(exact - numerical_s_matrix(x, steps)))
        errors.append(max(differences))
    if errors[1] > 2e-8 or abs(scattering(xr)[0] - math.pi/2) > 1e-10:
        raise ArithmeticError('径向积分或共振点未通过核验')
    return errors


# 以下两图使用内嵌SVG坐标与系统字体，不需要作图库或网络服务。
def line(x1, y1, x2, y2, color, width=1.5, dash=''):
    style = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}"{style}/>')


def label(x, y, value, size=18, color=INK, align='start', weight=400):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{align}" font-weight="{weight}">{escape(str(value))}</text>')


def curve(points, color, width=3, dash=''):
    style = f' stroke-dasharray="{dash}"' if dash else ''
    coords = ' '.join(f'{x:.2f},{y:.2f}' for x,y in points)
    return (f'<polyline points="{coords}" stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round" fill="none"{style}/>')


def svg_file(path, height, title, items):
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" '
            f'viewBox="0 0 1100 {height}"><title>{escape(title)}</title>'
            '<rect width="100%" height="100%" fill="#ffffff"/>'
            '<g font-family="Segoe UI, Microsoft YaHei, Noto Sans CJK SC, sans-serif">')
    path.write_text(head + '\n' + '\n'.join(items) + '\n</g></svg>\n', encoding='utf-8')


def scaled(data_lo, data_hi, page_lo, page_hi):
    return lambda value: page_lo + (value-data_lo)*(page_hi-page_lo)/(data_hi-data_lo)


def axes(items, xmap, ymap, xticks, yticks, box, show_x=True):
    x0, top, x1, bottom = box
    for val, name in xticks:
        xx = xmap(val)
        items.append(line(xx, top, xx, bottom, GRID, 1))
        if show_x:
            items.append(label(xx, bottom+23, name, 16, align='middle'))
    for val, name in yticks:
        yy = ymap(val)
        items.extend([line(x0, yy, x1, yy, GRID, 1),
                      label(x0-12, yy+5, name, 16, align='end')])
    items.extend([line(x0, top, x0, bottom, INK), line(x0, bottom, x1, bottom, INK)])


def plot_scattering(folder, data, xr):
    title = '原题6.13 · 球形方阱的 f 波共振'
    items = [label(35,47,title,34,weight=700),
             label(36,82,'图6.15复现　l = 3，阱深 30.25 E_R，横轴为 kR',21)]
    xl, xh = data[0][0], data[-1][0]
    xx = scaled(xl,xh,104,1045)
    xticks = [(x,f'{x:.2f}') for x in [1.32,1.35,1.40,1.45,1.50]]
    for n,(top,bottom,maxy,yticks,heading) in enumerate([
        (151,382,55,range(0,51,10),'(a)　分波截面 σ₃ / R²'),
        (480,706,math.pi,[0,0.5,1,1.5,2,2.5,3],'(b)　散射相移 δ₃（弧度）')]):
        yy = scaled(0,maxy,bottom,top)
        items.append(label(42,top-19,heading,23,weight=600))
        axes(items,xx,yy,xticks,[(y,f'{y:g}') for y in yticks],(104,top,1045,bottom),n==1)
        items.append(line(xx(xr),top,xx(xr),bottom,ORANGE,1.8,'4 5'))
        if n==0:
            items.append(curve([(xx(x),yy(bound)) for x,phase,cross,bound in data],GREEN,2,'7 5'))
            items.append(curve([(xx(x),yy(cross)) for x,phase,cross,bound in data],BLUE,3.3))
            items.extend([line(702,114,744,114,GREEN,2,'7 5'),label(753,120,'幺正上限 28π/(kR)²',17,GREEN)])
        else:
            items.extend([line(104,yy(math.pi/2),xx(xr),yy(math.pi/2),ORANGE,1.4,'3 4'),
                          label(112,yy(math.pi/2)-8,'π/2',17,ORANGE)])
            items.append(curve([(xx(x),yy(phase)) for x,phase,cross,bound in data],BLUE,3.3))
    items.extend([label(1045,759,'kR',20,align='end'),
                  label(38,798,f'相移穿过 π/2 时，kR = {xr:.6f}；两面板用同一条垂线定位。',20)])
    svg_file(folder/(STEM+'截面与相移.svg'),820,title,items)


def plot_potential(folder, xr):
    title='原题6.13 · 同一方阱中的能量与离心势垒'
    x0,x1,top,bottom=102,1045,169,653
    xs=scaled(0,3.5,x0,x1); ys=scaled(-34,24,bottom,top)
    ll=ELL*(ELL+1); er=xr*xr
    near=math.sqrt(ll/(DEPTH+er));far=math.sqrt(ll/er)
    items=[label(35,47,title,34,weight=700),
           label(36,82,'图6.14的方阱实例　半径以 R 为单位，能量以 E_R = ℏ²/(2mR²) 为单位',20),
           label(36,130,'势或能量 / E_R',20,weight=600)]
    axes(items,xs,ys,[(x,f'{x:g}') for x in [0,0.5,1,1.5,2,2.5,3,3.5]],
         [(y,str(y)) for y in [-30,-20,-10,0,10,20]],(x0,top,x1,bottom))
    # 深度和势垒使用同一能量零点。外势垒的阴影仅标注 E < V_eff 的区段。
    ext=[(1+(far-1)*i/160) for i in range(161)]
    fill=[(xs(1),ys(er))]+[(xs(r),ys(ll/r**2)) for r in ext]+[(xs(far),ys(er))]
    coords=' '.join(f'{x:.2f},{y:.2f}' for x,y in fill)
    items.append(f'<polygon points="{coords}" fill="#dcefe7"/>')
    items.extend([line(xs(0),ys(-DEPTH),xs(1),ys(-DEPTH),GREEN,3,'10 5'),
                  line(xs(1),ys(-DEPTH),xs(1),ys(0),GREEN,1.8,'5 5'),
                  line(xs(1),ys(0),xs(3.5),ys(0),GREEN,2.5,'10 5')])
    start=math.sqrt(ll/(24+DEPTH))
    rin=[start+(1-start)*i/330 for i in range(331)]
    rout=[1+2.5*i/550 for i in range(551)]
    items.extend([curve([(xs(r),ys(ll/r**2-DEPTH)) for r in rin],BLUE,3.5),
                  curve([(xs(r),ys(ll/r**2)) for r in rout],BLUE,3.5),
                  line(xs(1),ys(ll-DEPTH),xs(1),ys(ll),BLUE,1.6,'2 4'),
                  line(x0,ys(er),x1,ys(er),ORANGE,2.6,'9 4')])
    for r in (near,far):
        items.append(f'<circle cx="{xs(r):.2f}" cy="{ys(er):.2f}" r="4.5" fill="{ORANGE}"/>')
    items.extend([label(xs(0.60),ys(21),'内侧趋于 +∞',16,BLUE),
                  label(xs(1.04),ys(13.4),'12',17,BLUE),
                  label(xs(1.05),ys(-18.25)+6,'−18.25',17,BLUE),
                  label(xs(0.10),ys(-DEPTH)-9,'井底 −30.25',17,GREEN),
                  label(xs(2.5),ys(er)-11,f'共振能 {er:.4f}',17,ORANGE),
                  label(xs(1.66),ys(-6),'外侧禁阻区',17,GREEN),
                  line(xs(1.84),ys(-4),xs(1.55),ys(3.0),GREEN,1.1),
                  label(x1,bottom+51,'r / R',20,align='end'),
                  label(38,736,f'转向点 r/R = {near:.4f} 与 {far:.4f}；阱边界在 r/R = 1。',20),
                  label(38,771,'蓝线为有效势，绿色虚线为裸方阱，橙线为散射能量。',20)])
    svg_file(folder/(STEM+'能量阱深与有效势.svg'),795,title,items)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    folder=parser.parse_args().output_dir
    folder.mkdir(parents=True,exist_ok=True)
    xr, peak, width, left, right = resonance()
    errors = check_results(xr)
    nodes = sorted({1.31 + 0.21*i/1050 for i in range(1051)} | {xr,peak,left,right})
    data = [(x,*scattering(x)) for x in nodes]
    with (folder/(STEM+'共振曲线.csv')).open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['kR','delta3_rad','sigma3_over_R2','unitarity_limit_over_R2'])
        writer.writerows(data)
    plot_scattering(folder,data,xr)
    plot_potential(folder,xr)
    print(f'x_r={xr:.11f}, E_r/E_R={xr*xr:.11f}, sigma_r/R^2={scattering(xr)[1]:.9f}')
    print(f'x_peak={peak:.11f}, sigma_peak/R^2={scattering(peak)[1]:.9f}')
    print(f'Gamma/E_R={width:.10f}, half-height roots={left:.10f}, {right:.10f}')
    print(f'energy-FWHM/E_R={right**2-left**2:.10f}, max |delta S| at 512/1024 steps: {errors}')


if __name__ == '__main__':
    main()
