"""
DelFly Explorer — V-Rocker 4-Bar  Stress & Strain Map  v10
FULL SCRIPT (verbatim from user, only SAVE path changed for Linux testing)
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Arc

SAVE = r"/Users/apple/Visual studio code/Flappers/Saves"
os.makedirs(SAVE, exist_ok=True)

dx_val      = 12.5
l1_val      = 10.0
target_down = -50.0
bend_deg    = 165.0
y2_val      = 20.0

arm_w, arm_t = 3.0, 2.0   # FIX: thickness 1.5->2.0mm to clear yield
cr_w,  cr_t  = 3.0, 1.8   # FIX: thickness 1.2->1.8mm to clear yield
co_w,  co_t  = 3.0, 1.2
ca_w,  ca_t  = 3.0, 2.0

T_tip_right   = 50.0
T_tip_left    = 50.0

n_sides       = 2
motor_rpm     = 43_000
gearbox_ratio = 35.94
eta           = 0.85
E_pla, sy, su = 3500., 45., 55.

def get_l2(dx, l1, y, dy): return np.sqrt((y - dx) ** 2 + (dy - l1) ** 2)

def calc_phi(dx, l1, th_deg, y, dy, l2):
    th = np.radians(th_deg); bx, by = l1 * np.cos(th), l1 * np.sin(th)
    ax, ay = -dx, dy; d = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    cv = (y ** 2 + d ** 2 - l2 ** 2) / (2 * y * d)
    if abs(cv) > 1: return None
    return np.degrees(np.arctan2(by - ay, bx - ax) + np.arccos(np.clip(cv, -1, 1)))

def calc_gamma(dx, l1, th_deg, y, dy, l2):
    th = np.radians(th_deg); bx, by = l1 * np.cos(th), l1 * np.sin(th)
    ax, ay = -dx, dy; d = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    return np.degrees(np.arccos(np.clip((l2 ** 2 + y ** 2 - d ** 2) / (2 * l2 * y), -1, 1)))

def phi_arr(dx, l1, y, dy, l2, n=720):
    th = np.linspace(0, 360, n, endpoint=False)
    r = [calc_phi(dx, l1, t, y, dy, l2) for t in th]
    return th, np.array([v if v is not None else np.nan for v in r])

def solve(dx, l1, target, gn=80, tn=180):
    yr = (1.2 * l1, 3.0 * l1); dr = (2.0 * dx, 3.6 * dx)
    best, berr = None, 1e9
    ths = np.linspace(0, 360, tn)
    for ty in np.linspace(*yr, gn):
        for td in np.linspace(*dr, gn):
            l2 = get_l2(dx, l1, ty, td)
            _, ph = phi_arr(dx, l1, ty, td, l2, n=tn)
            if np.any(np.isnan(ph)): continue
            e = abs(ph.max()) + abs(ph.min() - target)
            if e < berr: berr = e; best = (ty, td, l2, ths[np.argmax(ph)], ths[np.argmin(ph)])
    return best

def cross2(a, b): return a[0] * b[1] - a[1] * b[0]

print("Solving linkage...")
y, dy, l2, *_ = solve(dx_val, l1_val, target_down)
print(f"  y={y:.3f}  dy={dy:.3f}  l2={l2:.3f} mm")

ths = np.linspace(0, 360, 3600)
gams = np.array([calc_gamma(dx_val, l1_val, t, y, dy, l2) for t in ths])
tw = ths[np.argmin(gams)]; gw = gams.min()
pw = calc_phi(dx_val, l1_val, tw, y, dy, l2)
print(f"  Worst theta={tw:.2f} deg  gamma={gw:.2f} deg  phi={pw:.2f} deg")

pr = np.radians(pw); pEr = pr + np.radians(bend_deg); tr = np.radians(tw)
Bx = l1_val * np.cos(tr);  By = l1_val * np.sin(tr)
Ax = -dx_val;              Ay = dy
Cx = Ax + y * np.cos(pr);  Cy = Ay + y * np.sin(pr)
Ex = Ax + y * np.cos(pEr); Ey = Ay + y * np.sin(pEr)
D1x = Cx + y2_val; D1y = Cy
D2x = Ex - y2_val; D2y = Ey

Bm = np.array([Bx, By]) * 1e-3; Am = np.array([Ax, Ay]) * 1e-3
Cm = np.array([Cx, Cy]) * 1e-3
cv = Bm - Cm; cd = cv / np.linalg.norm(cv)
ACm = Cm - Am; ma = abs(cross2(ACm, cd))

T_right = T_tip_right * 1e-3
T_left  = T_tip_left  * 1e-3
M_A = T_right + T_left
Fc = M_A / ma

u_OB = Bm / np.linalg.norm(Bm)
Fc_vec = Fc * cd
Tc1 = abs(cross2(Bm, Fc_vec))
Fax_crank = abs(np.dot(Fc_vec, u_OB))
Tcs = n_sides * Tc1
Tm = Tcs / (gearbox_ratio * eta)

ad = np.array([np.cos(pr), np.sin(pr)])
Fax = Fc * abs(np.dot(cd, ad))

print(f"  Coupler force={Fc:.2f} N   Crank torque={Tc1*1e3:.2f} mN*m   "
      f"Shaft (x{n_sides})={Tcs*1e3:.2f} mN*m   Motor={Tm*1e6:.1f} uN*m")

def rect(w, t): return w * t, w * t ** 3 / 12, t / 2

def stress(Fax, Mr, Mt, L, w, t, n=400):
    A, I, c = rect(w, t); s = np.linspace(0, 1, n)
    M = (Mr + s * (Mt - Mr)) * 1e3
    return s, np.abs(M * c / I) + abs(Fax / A)

s_cr, sig_cr = stress(Fax_crank, Tc1, 0.0, l1_val, cr_w, cr_t)
s_co, sig_co = stress(Fc, 0.0, 0.0, l2, co_w, co_t)
M_at_A_right = T_right - M_A;  M_at_C = T_right
s_a1, sig_a1 = stress(Fax, M_at_A_right, M_at_C, y, arm_w, arm_t)
M_at_A_left = T_left - M_A;    M_at_E = T_left
s_a2, sig_a2 = stress(Fax, M_at_A_left, M_at_E, y, arm_w, arm_t)
s_ca_r, sig_ca_r = stress(0.0, T_right, T_right, y2_val, ca_w, ca_t)
s_ca_l, sig_ca_l = stress(0.0, T_left, T_left, y2_val, ca_w, ca_t)

BG = '#0b0f1c'
_cmap = LinearSegmentedColormap.from_list('sv8', [
    (.04, .06, .28), (.08, .38, .78), (.00, .78, .92), (.18, .92, .28),
    (.96, .88, .08), (1., .50, .00), (.92, .08, .08), (1., 1., 1.)], N=512)

all_sig = np.concatenate([sig_cr, sig_co, sig_a1, sig_a2, sig_ca_r, sig_ca_l])
vmax = max(su * 1.2, all_sig.max() * 1.02)
NRM = Normalize(0, vmax)

fig = plt.figure(figsize=(18, 9), facecolor=BG)
ax  = fig.add_axes([0.02, 0.06, 0.55, 0.86])
cax = fig.add_axes([0.585, 0.06, 0.016, 0.86])
axr = fig.add_axes([0.62, 0.06, 0.37, 0.86])
ax.set_facecolor(BG); axr.set_facecolor(BG); axr.axis('off')

def lc(ax, s, sig, x0, y0, x1, y1, lw=7):
    pts = np.c_[np.interp(s, [0, 1], [x0, x1]), np.interp(s, [0, 1], [y0, y1])].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    col = LineCollection(segs, colors=_cmap(NRM(sig[:-1])), linewidth=lw, capstyle='round', zorder=3)
    ax.add_collection(col)
    fail = np.where(sig > sy)[0]
    for fi in fail[::max(1, len(fail) // 5)]:
        ax.plot(np.interp(s[fi], [0, 1], [x0, x1]), np.interp(s[fi], [0, 1], [y0, y1]),
                 'w+', ms=9, mew=2, zorder=10)

lc(ax, s_cr,   sig_cr,   0, 0,  Bx, By,  lw=9)
lc(ax, s_co,   sig_co,   Bx, By, Cx, Cy, lw=7)
lc(ax, s_a1,   sig_a1,   Ax, Ay, Cx, Cy, lw=11)
lc(ax, s_a2,   sig_a2,   Ax, Ay, Ex, Ey, lw=11)
lc(ax, s_ca_r, sig_ca_r, Cx, Cy, D1x, D1y, lw=7)
lc(ax, s_ca_l, sig_ca_l, Ex, Ey, D2x, D2y, lw=7)

arc_r = 6.0
ax.add_patch(Arc((Ax, Ay), 2 * arc_r, 2 * arc_r, angle=0,
                  theta1=np.degrees(pr), theta2=np.degrees(pEr), color='#8888ff', lw=1.5, zorder=5))
mid_a = (np.degrees(pr) + np.degrees(pEr)) / 2
ax.text(Ax + (arc_r + 3) * np.cos(np.radians(mid_a)), Ay + (arc_r + 3) * np.sin(np.radians(mid_a)),
        f'{bend_deg:.0f}\u00b0', color='#8888ff', fontsize=9, ha='center', va='center', fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])

TR = 4.5

def torque_sym(ax, cx, cy, ccw, color, label, label_dx=0, label_dy=-7):
    if ccw:
        t1, t2 = 30, 310; ah_from = np.radians(316); ah_to = np.radians(304)
    else:
        t1, t2 = 30, 310; ah_from = np.radians(34); ah_to = np.radians(46)
    ax.add_patch(Arc((cx, cy), 2 * TR, 2 * TR, angle=0, theta1=t1, theta2=t2, color=color, lw=2.0, zorder=7))
    ax.annotate('', xy=(cx + TR * np.cos(ah_to), cy + TR * np.sin(ah_to)),
                 xytext=(cx + TR * np.cos(ah_from), cy + TR * np.sin(ah_from)),
                 arrowprops=dict(arrowstyle='->', color=color, lw=2.0, mutation_scale=12), zorder=8)
    ax.text(cx + label_dx, cy + label_dy, label, color=color, fontsize=8, ha='center', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])

torque_sym(ax, D1x, D1y, ccw=True,  color='#00ccff',
           label=f'+{T_tip_right:.1f} mN\u00b7m\n(CCW)', label_dx=3,  label_dy=-8)
torque_sym(ax, D2x, D2y, ccw=False, color='#ff5555',
           label=f'\u2212{T_tip_left:.1f} mN\u00b7m\n(CW)',  label_dx=-4, label_dy=-8)
torque_sym(ax, 0, 0, ccw=True, color='#ff9900',
           label=f'Motor shaft\n{Tcs*1e3:.0f} mN\u00b7m', label_dx=0, label_dy=TR + 3)

def joint(ax, x, y, fc, label, lx, ly, r=1.4):
    ax.add_patch(plt.Circle((x, y), r, color=fc, zorder=11))
    ax.add_patch(plt.Circle((x, y), r * 2, fill=False, ec=fc, lw=1.3, zorder=10))
    ax.text(x + lx, y + ly, label, color=fc, fontsize=7.5, va='center',
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)], zorder=12)

joint(ax, 0,  0,  '#ffc400', 'O (crank/gear pivot)', 1.5, -3.5)
joint(ax, Bx, By, '#ffffff', 'B (crank pin)',         1.5,  1.5)
joint(ax, Ax, Ay, '#aaaaaa', 'A (base pivot)',         1.5, -3.5)
joint(ax, Cx, Cy, '#dddddd', 'C (coupler pin)',        1.5,  1.5)
joint(ax, Ex, Ey, '#999999', 'E (left arm tip)',      -18,   1.5)

ax.add_patch(plt.Circle((0, 0), 3, fill=False, ec='#2255aa', lw=1.3, ls='--', zorder=2))
ax.text(0, -5.5, f'GR={gearbox_ratio}:1', color='#4477cc', fontsize=7.5, ha='center',
        path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

hw = 5
for gx in np.linspace(-hw, hw, 7):
    ax.plot([Ax + gx, Ax + gx - 1.8], [Ay - 1.5, Ay - 4.5], color='#445566', lw=1)
ax.plot([Ax - hw - 1, Ax + hw + 1], [Ay - 1.5, Ay - 1.5], color='#778899', lw=1.8)

def llab(ax, x0, y0, x1, y1, txt, dx=0, dy=3, fs=7.8):
    ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, txt, color='#cccccc', fontsize=fs, ha='center',
            style='italic', path_effects=[pe.withStroke(linewidth=2.2, foreground=BG)])

llab(ax, 0,  0,  Bx, By, f'Crank {l1_val:.0f}mm',      dx=2,  dy=2)
llab(ax, Bx, By, Cx, Cy, f'Coupler {l2:.1f}mm',         dx=0,  dy=-4)
llab(ax, Ax, Ay, Cx, Cy, f'Right arm {y:.1f}mm',        dx=1,  dy=-5)
llab(ax, Ax, Ay, Ex, Ey, f'Left arm {y:.1f}mm',         dx=6,  dy=2)
llab(ax, Cx, Cy, D1x, D1y, 'Right spar \u2192',         dx=0,  dy=3)
llab(ax, Ex, Ey, D2x, D2y, '\u2190 Left spar',          dx=0,  dy=3)

all_x = [0, Bx, Ax, Cx, Ex, D1x, D2x]; all_y = [0, By, Ay, Cy, Ey, D1y, D2y]
xm = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 0.15
ax.set_xlim(min(all_x) - xm - 8,  max(all_x) + xm + 12)
ax.set_ylim(min(all_y) - xm - 10, max(all_y) + xm + 8)
ax.set_aspect('equal')
ax.set_xlabel('X (mm)', color='#888', fontsize=9)
ax.set_ylabel('Y (mm)', color='#888', fontsize=9)
ax.tick_params(colors='#666')
for sp in ax.spines.values(): sp.set_edgecolor('#1e2535')
ax.grid(True, color='#ffffff07', lw=0.5)

sm = plt.cm.ScalarMappable(cmap=_cmap, norm=NRM); sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Stress (MPa)', color='white', fontsize=9, labelpad=6)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)
cbar.ax.axhline(sy, color='#ff3333', lw=2,   ls='--')
cbar.ax.axhline(su, color='#ff9900', lw=1.3, ls=':')
cbar.ax.text(4.0, sy, f' Yield\n {sy}', color='#ff4444', fontsize=7.5,
             transform=cbar.ax.transData, va='center')
cbar.ax.text(4.0, su, f' UTS\n {su}',   color='#ff9900', fontsize=7.5,
             transform=cbar.ax.transData, va='center')
cbar.ax.set_facecolor(BG)

def ptitle(ax, y, txt, c='#4488ff'):
    ax.text(0.0, y, txt, transform=ax.transAxes, color=c, fontsize=9.5, fontweight='bold', va='top')
    return y - 0.040

def prow(ax, y, lbl, val, vc='#aaff88', fs=8.8):
    ax.text(0.02, y, lbl, transform=ax.transAxes, color='#999999', fontsize=fs, va='top', fontfamily='monospace')
    ax.text(0.58, y, val, transform=ax.transAxes, color=vc,        fontsize=fs, va='top', fontfamily='monospace')
    return y - 0.038

def psep(ax, y):
    ax.plot([0, 1], [y - 0.006, y - 0.006], transform=ax.transAxes, color='#1e2d44', lw=0.8)
    return y - 0.022

y0 = 0.98
y0 = ptitle(axr, y0, 'SOLVED LINKAGE GEOMETRY')
y0 = prow(axr, y0, 'Arm length y',     f'{y:.4f} mm')
y0 = prow(axr, y0, 'Pivot height dy',  f'{dy:.4f} mm')
y0 = prow(axr, y0, 'Coupler l2',       f'{l2:.4f} mm')
y0 = prow(axr, y0, 'V-bend angle',     f'{bend_deg:.0f}\u00b0')
y0 = prow(axr, y0, 'Spar length',      f'{y2_val:.0f} mm (each)')
y0 = psep(axr, y0)
y0 = ptitle(axr, y0, 'KINEMATICS')
y0 = prow(axr, y0, 'dx, l1',           f'{dx_val} / {l1_val} mm')
y0 = prow(axr, y0, 'Worst crank theta',f'{tw:.2f}\u00b0')
y0 = prow(axr, y0, 'Trans. angle gamma',f'{gw:.2f}\u00b0')
y0 = prow(axr, y0, 'phi_body (worst)', f'{pw:.2f}\u00b0')
y0 = prow(axr, y0, 'Flap frequency',   f'{motor_rpm/gearbox_ratio/60:.2f} Hz')
y0 = psep(axr, y0)
y0 = ptitle(axr, y0, 'TORQUE DIRECTIONS')
y0 = prow(axr, y0, 'Right spar', f'+{T_tip_right:.1f} mN\u00b7m CCW', vc='#00ccff')
y0 = prow(axr, y0, 'Left spar',  f'\u2212{T_tip_left:.1f} mN\u00b7m CW',  vc='#ff5555')
y0 = prow(axr, y0, 'Net at pivot A', f'{M_A*1e3:.1f} mN\u00b7m',          vc='#ffffaa')
y0 = psep(axr, y0)
y0 = ptitle(axr, y0, 'TORQUE CHAIN (through gearbox)')
y0 = prow(axr, y0, 'Coupler force',        f'{Fc:.3f} N')
y0 = prow(axr, y0, 'Crank torque (x1)',    f'{Tc1*1e3:.3f} mN\u00b7m')
y0 = prow(axr, y0, f'Shaft (x{n_sides} cranks)', f'{Tcs*1e3:.3f} mN\u00b7m')
y0 = prow(axr, y0, f'Motor (eta={eta})',   f'{Tm*1e6:.1f} uN\u00b7m')
y0 = prow(axr, y0, 'Gear ratio',           f'{gearbox_ratio}:1')
y0 = psep(axr, y0)
y0 = ptitle(axr, y0, f'STRESS & STRAIN AT THIS POSE  (PLA  sigma_y={sy} MPa)')

axr.text(0.02, y0, f'{"Link":<16}  {"sigma_peak":>10}  {"strain":>9}',
         transform=axr.transAxes, color='#666677', fontsize=8.5, va='top', fontfamily='monospace')
y0 -= 0.030

for nm, sig_arr in [('Crank', sig_cr), ('Coupler', sig_co), ('Right arm', sig_a1),
                    ('Left arm', sig_a2), ('Right spar', sig_ca_r), ('Left spar', sig_ca_l)]:
    pk = sig_arr.max(); eps = pk / E_pla
    vc  = '#ff4444' if pk > sy else '#aaff88'
    sym = '\u2717'  if pk > sy else '\u2713'
    axr.text(0.02, y0, f'{nm:<16}  {pk:>8.2f}  {eps:>9.5f} {sym}',
              transform=axr.transAxes, color=vc, fontsize=8.5, va='top', fontfamily='monospace')
    y0 -= 0.036

y0 = psep(axr, y0)
axr.text(0.02, y0 - 0.005,
          'Note: this panel = ONE pose (worst gamma).\n'
          'See section 6 sweep figure + console table\n'
          'for the true worst-case angle per member\n'
          'across the full 360\u00b0 crank rotation.',
          transform=axr.transAxes, color='#ffcc44', fontsize=8.2, va='top',
          bbox=dict(fc='#19130a', ec='#554400', boxstyle='round,pad=0.5'))

for sp in axr.spines.values():
    sp.set_visible(True); sp.set_edgecolor('#1a2535'); sp.set_lw(1)

fig.suptitle(
    'DelFly Explorer  \u00b7  V-Rocker 4-Bar  \u00b7  Stress & Strain Map  v10\n'
    f'Solved: y={y:.2f} mm  l2={l2:.2f} mm  |  {bend_deg:.0f}\u00b0 V-bend  |  PLA  |  '
    f'T_R={T_tip_right:.1f} mN\u00b7m (CCW)  T_L={T_tip_left:.1f} mN\u00b7m (CW)',
    color='white', fontsize=10.5, fontweight='bold', y=0.985)

out = os.path.join(SAVE, 'v_rocker_stress_v10.png')
plt.savefig(out, dpi=185, bbox_inches='tight', facecolor=BG)
print(f"Saved -> {out}")
plt.close()

print(f"\n{'='*60}")
print(f"  SINGLE-POSE (worst-gamma) RESULT — for reference only")
print(f"  theta_worst={tw:.2f}  gamma={gw:.2f}")
print(f"  Coupler={Fc:.3f}N  T_crank={Tc1*1e3:.3f}mNm  "
      f"T_shaft={Tcs*1e3:.3f}mNm  T_motor={Tm*1e6:.1f}uNm")
print(f"  {'Link':<14} {'sigma_peak':>12} {'strain':>12}")
print(f"  {'-'*40}")
for nm, sig_arr in [('Crank', sig_cr), ('Coupler', sig_co), ('Right arm', sig_a1),
                    ('Left arm', sig_a2), ('Right spar', sig_ca_r), ('Left spar', sig_ca_l)]:
    pk = sig_arr.max(); eps = pk / E_pla
    print(f"  {nm:<14} {pk:>12.2f} {eps:>12.6f}"
          f"{'  <- EXCEEDS YIELD' if pk > sy else ''}")
print(f"{'='*60}")

def pose_full(theta_deg):
    phi = calc_phi(dx_val, l1_val, theta_deg, y, dy, l2)
    if phi is None:
        return None
    pr_  = np.radians(phi)
    pEr_ = pr_ + np.radians(bend_deg)
    tr_  = np.radians(theta_deg)
    Bx_ = l1_val * np.cos(tr_);   By_ = l1_val * np.sin(tr_)
    Ax_ = -dx_val;                Ay_ = dy
    Cx_ = Ax_ + y * np.cos(pr_);  Cy_ = Ay_ + y * np.sin(pr_)
    Ex_ = Ax_ + y * np.cos(pEr_); Ey_ = Ay_ + y * np.sin(pEr_)
    Bm_ = np.array([Bx_, By_]) * 1e-3
    Am_ = np.array([Ax_, Ay_]) * 1e-3
    Cm_ = np.array([Cx_, Cy_]) * 1e-3
    cv_ = Bm_ - Cm_
    nrm = np.linalg.norm(cv_)
    if nrm < 1e-12: return None
    cd_ = cv_ / nrm
    ACm_ = Cm_ - Am_; ma_ = abs(cross2(ACm_, cd_))
    if ma_ < 1e-9: return None
    T_right_ = T_tip_right * 1e-3
    T_left_  = T_tip_left  * 1e-3
    M_A_ = T_right_ + T_left_
    Fc_  = M_A_ / ma_
    u_OB_   = Bm_ / np.linalg.norm(Bm_)
    Fc_vec_ = Fc_ * cd_
    Tc1_        = abs(cross2(Bm_, Fc_vec_))
    Fax_crank_  = abs(np.dot(Fc_vec_, u_OB_))
    Tcs_ = n_sides * Tc1_
    Tm_  = Tcs_ / (gearbox_ratio * eta)
    ad_  = np.array([np.cos(pr_),  np.sin(pr_)])
    ad2_ = np.array([np.cos(pEr_), np.sin(pEr_)])
    Fax_right_ = Fc_ * abs(np.dot(cd_, ad_))
    Fax_left_  = Fc_ * abs(np.dot(cd_, ad2_))
    M_at_A_right_ = T_right_ - M_A_;  M_at_C_ = T_right_
    M_at_A_left_  = T_left_  - M_A_;  M_at_E_ = T_left_
    _, sig_cr_  = stress(Fax_crank_, Tc1_,         0.0,     l1_val, cr_w,  cr_t)
    _, sig_co_  = stress(Fc_,        0.0,           0.0,     l2,     co_w,  co_t)
    _, sig_a1_  = stress(Fax_right_, M_at_A_right_, M_at_C_, y,      arm_w, arm_t)
    _, sig_a2_  = stress(Fax_left_,  M_at_A_left_,  M_at_E_, y,      arm_w, arm_t)
    _, sig_car_ = stress(0.0, T_right_, T_right_, y2_val, ca_w, ca_t)
    _, sig_cal_ = stress(0.0, T_left_,  T_left_,  y2_val, ca_w, ca_t)
    return dict(
        theta=theta_deg, phi=phi,
        Fc=Fc_, Tc1=Tc1_, Tcs=Tcs_, Tm=Tm_,
        sig_crank=sig_cr_.max(),    eps_crank=sig_cr_.max()/E_pla,
        sig_coupler=sig_co_.max(),  eps_coupler=sig_co_.max()/E_pla,
        sig_arm_r=sig_a1_.max(),    eps_arm_r=sig_a1_.max()/E_pla,
        sig_arm_l=sig_a2_.max(),    eps_arm_l=sig_a2_.max()/E_pla,
        sig_spar_r=sig_car_.max(),  eps_spar_r=sig_car_.max()/E_pla,
        sig_spar_l=sig_cal_.max(),  eps_spar_l=sig_cal_.max()/E_pla,
    )

print("\nSweeping ALL crank angles theta (0-360 deg, 0=vertical up CW+)...")
N_SWEEP = 720
your_sweep   = np.linspace(0, 360, N_SWEEP, endpoint=False)
thetas_sweep = (90.0 - your_sweep) % 360.0
rows = [pose_full(t) for t in thetas_sweep]
rows = [r for r in rows if r is not None]
print(f"  Valid poses: {len(rows)}/{N_SWEEP}")

member_keys  = ['sig_crank',  'sig_coupler',  'sig_arm_r',  'sig_arm_l',  'sig_spar_r',  'sig_spar_l']
strain_keys  = ['eps_crank',  'eps_coupler',  'eps_arm_r',  'eps_arm_l',  'eps_spar_r',  'eps_spar_l']
member_names = ['Crank',      'Coupler',      'Right arm',  'Left arm',   'Right spar',  'Left spar']

th_v   = your_sweep[:len(rows)]
Tc1_v  = np.array([r['Tc1']  for r in rows]) * 1e3
Tcs_v  = np.array([r['Tcs']  for r in rows]) * 1e3
Tm_v   = np.array([r['Tm']   for r in rows]) * 1e6
Fc_v   = np.array([r['Fc']   for r in rows])

sig_sweep = {k: np.array([r[k] for r in rows]) for k in member_keys}
eps_sweep = {k: np.array([r[k] for r in rows]) for k in strain_keys}

print(f"\n  {'Link':<14} {'sigma_max':>10} {'at theta (your conv.)':>22} {'strain_max':>12}")
print(f"  {'-'*62}")
worst_member, worst_val, worst_theta = None, -1, None
for nm, sk, ek in zip(member_names, member_keys, strain_keys):
    arr = sig_sweep[sk]; i = np.argmax(arr)
    flag = '  <- EXCEEDS YIELD' if arr[i] > sy else ''
    print(f"  {nm:<14} {arr[i]:>10.2f} {th_v[i]:>21.1f}\u00b0  {arr[i]/E_pla:>10.5f}{flag}")
    if arr[i] > worst_val:
        worst_val, worst_member, worst_theta = arr[i], nm, th_v[i]

print(f"  {'-'*62}")
print(f"  GLOBAL WORST: {worst_member} at theta={worst_theta:.1f}\u00b0 (your conv.), "
      f"sigma={worst_val:.2f} MPa, strain={worst_val/E_pla:.5f}")
print(f"  Shaft torque range : {Tcs_v.min():.3f} \u2013 {Tcs_v.max():.3f} mN\u00b7m")
print(f"  Motor torque range : {Tm_v.min():.2f}  \u2013 {Tm_v.max():.2f}  \u00b5N\u00b7m")
print(f"  Coupler force range: {Fc_v.min():.3f} \u2013 {Fc_v.max():.3f} N")

colors6 = ['#ff9900', '#00ccff', '#ff5555', '#aa88ff', '#66ee44', '#ffee44']

fig2, axes = plt.subplots(2, 2, figsize=(18, 11), facecolor=BG)
fig2.subplots_adjust(hspace=0.38, wspace=0.30, left=0.07, right=0.97, top=0.90, bottom=0.07)

def style_ax(a):
    a.set_facecolor(BG)
    a.tick_params(colors='#aaa', labelsize=8.5)
    a.set_xlabel('Crank angle \u03b8 (deg,  0\u00b0=vertical up, CW+)', color='#aaa', fontsize=9)
    a.grid(True, color='white', alpha=0.10, lw=0.5)
    for sp in a.spines.values(): sp.set_edgecolor('#1e2535')
    a.set_xlim(0, 360)
    a.set_xticks(np.arange(0, 361, 45))

ax1 = axes[0, 0]; style_ax(ax1)
for nm, sk, c in zip(member_names, member_keys, colors6):
    ax1.plot(th_v, sig_sweep[sk], color=c, lw=1.8, label=nm)
ax1.axhline(sy, color='#ff3333', lw=1.5, ls='--', label=f'Yield  {sy:.0f} MPa')
ax1.axhline(su, color='#ff7700', lw=1.2, ls=':',  label=f'UTS    {su:.0f} MPa')
ax1.set_ylabel('Peak stress (MPa)', color='#ccc', fontsize=9)
ax1.set_title('Stress of every member vs crank angle',
              color='white', fontsize=10.5, fontweight='bold', pad=7)
ax1.legend(facecolor='#0d1628', edgecolor='#1e2535', fontsize=7.5,
           labelcolor='white', ncol=2, loc='upper right')
for nm, sk, c in zip(member_names, member_keys, colors6):
    arr = sig_sweep[sk]; i = np.argmax(arr)
    ax1.plot(th_v[i], arr[i], 'o', color=c, ms=5, zorder=10)
    if arr[i] > sy:
        ax1.annotate(f'{nm}\n{arr[i]:.1f}', xy=(th_v[i], arr[i]),
                     xytext=(th_v[i] + 8, arr[i] + 1.5), color=c, fontsize=6.5,
                     arrowprops=dict(arrowstyle='->', color=c, lw=0.8))

ax2 = axes[0, 1]; style_ax(ax2)
for nm, ek, c in zip(member_names, strain_keys, colors6):
    ax2.plot(th_v, eps_sweep[ek] * 1e3, color=c, lw=1.8, label=nm)
yield_ms = (sy / E_pla) * 1e3
uts_ms   = (su / E_pla) * 1e3
ax2.axhline(yield_ms, color='#ff3333', lw=1.5, ls='--', label=f'Yield \u03b5  {yield_ms:.2f} m\u03b5')
ax2.axhline(uts_ms,   color='#ff7700', lw=1.2, ls=':',  label=f'UTS \u03b5    {uts_ms:.2f} m\u03b5')
ax2.set_ylabel('Peak strain (milli-strain \u00d710\u207b\u00b3)', color='#ccc', fontsize=9)
ax2.set_title('Strain of every member vs crank angle',
              color='white', fontsize=10.5, fontweight='bold', pad=7)
ax2.legend(facecolor='#0d1628', edgecolor='#1e2535', fontsize=7.5,
           labelcolor='white', ncol=2, loc='upper right')

ax3 = axes[1, 0]; style_ax(ax3)
ax3.plot(th_v, Tc1_v, color='#44aaff', lw=1.8, label='Crank torque (\u00d71),  mN\u00b7m')
ax3.plot(th_v, Tcs_v, color='#00ffcc', lw=2.0, label=f'Shaft torque (\u00d7{n_sides} cranks),  mN\u00b7m')
ax3t = ax3.twinx(); ax3t.set_facecolor(BG)
for sp in ax3t.spines.values(): sp.set_edgecolor('#1e2535')
ax3t.plot(th_v, Tm_v, color='#ffaa00', lw=1.6, ls='--', label='Motor torque,  \u00b5N\u00b7m')
ax3t.set_ylabel('Motor torque (\u00b5N\u00b7m)', color='#ffaa00', fontsize=9)
ax3t.tick_params(axis='y', colors='#ffaa00', labelsize=8.5)
ax3.set_ylabel('Torque (mN\u00b7m)', color='#ccc', fontsize=9)
ax3.set_title('Crank / Shaft / Motor torque vs crank angle',
              color='white', fontsize=10.5, fontweight='bold', pad=7)
l1_, lb1_ = ax3.get_legend_handles_labels()
l2_, lb2_ = ax3t.get_legend_handles_labels()
ax3.legend(l1_ + l2_, lb1_ + lb2_, facecolor='#0d1628', edgecolor='#1e2535',
           fontsize=7.5, labelcolor='white', loc='upper right')

gam_v = np.array([calc_gamma(dx_val, l1_val, (90.0-t)%360.0, y, dy, l2) for t in th_v])
ax4 = axes[1, 1]; style_ax(ax4)
ax4.plot(th_v, Fc_v, color='#ff88cc', lw=2.0, label='Coupler force Fc (N)')
ax4.fill_between(th_v, 0, Fc_v, alpha=0.18, color='#ff88cc')
ax4r = ax4.twinx(); ax4r.set_facecolor(BG)
for sp in ax4r.spines.values(): sp.set_edgecolor('#1e2535')
ax4r.plot(th_v, gam_v, color='#88ff99', lw=1.4, ls='-.', label='Transmission angle \u03b3 (deg)')
ax4r.set_ylabel('Transmission angle \u03b3 (deg)', color='#88ff99', fontsize=9)
ax4r.tick_params(axis='y', colors='#88ff99', labelsize=8.5)
ax4r.axhline(gw, color='#88ff99', lw=0.8, ls=':', alpha=0.6)
tw_your = (90.0 - tw) % 360.0
ax4.axvline(tw_your, color='#ffee55', lw=1.0, ls=':', alpha=0.7, label=f'Worst \u03b3 @ \u03b8={tw_your:.1f}\u00b0 (your conv.)')
ax4.set_ylabel('Coupler force (N)', color='#ccc', fontsize=9)
ax4.set_title('Coupler force & transmission angle vs crank angle',
              color='white', fontsize=10.5, fontweight='bold', pad=7)
l1_, lb1_ = ax4.get_legend_handles_labels()
l2_, lb2_ = ax4r.get_legend_handles_labels()
ax4.legend(l1_ + l2_, lb1_ + lb2_, facecolor='#0d1628', edgecolor='#1e2535',
           fontsize=7.5, labelcolor='white', loc='upper right')

fig2.suptitle(
    'DelFly Explorer  \u00b7  V-Rocker 4-Bar  \u00b7  Full 0\u2013360\u00b0 Crank Sweep  v10\n'
    f'y={y:.2f} mm   l2={l2:.2f} mm   l1={l1_val} mm   dx={dx_val} mm   '
    f'V-bend={bend_deg:.0f}\u00b0   PLA  E={E_pla} MPa  \u03c3_y={sy} MPa\n'
    f'T_tip_R={T_tip_right:.1f} mN\u00b7m (CCW)   T_tip_L={T_tip_left:.1f} mN\u00b7m (CW)   '
    f'n_cranks={n_sides}   GR={gearbox_ratio}:1   \u03b7={eta}',
    color='white', fontsize=10, fontweight='bold', y=0.97)

out2 = os.path.join(SAVE, 'v_rocker_full_sweep_v10.png')
plt.savefig(out2, dpi=185, bbox_inches='tight', facecolor=BG)
print(f"Saved -> {out2}")
plt.close(fig2)
print("DONE OK")