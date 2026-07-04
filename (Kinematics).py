"""
4-Bar Crank-Rocker Linkage Solver — V-Rocker Ornithopter (UNIVERSAL VERSION, CW-90 FRAME FIX)
================================================================================================

WHAT'S DIFFERENT IN THIS FILE VS THE PREVIOUS ONE
--------------------------------------------------
The kinematics (calculate_phi, calculate_gamma, get_l2, output_arm_state) define
"rest" at an INTERNAL crank angle theta_internal = 90 deg, increasing
counter-clockwise (standard math convention). That part is correct and was
never the problem: TR is mathematically guaranteed to hit exactly zero at the
phi extrema, because TR = |dphi/dtheta| and the extrema are defined by
dphi/dtheta = 0.

The actual bug: every theta value that got *reported or plotted* (the sweep
in __main__, theta_plot, th_max/th_min in the printout) was the raw INTERNAL
theta, with no conversion to your physical crank frame (0 deg = rest,
increasing CLOCKWISE, which is how you actually measure/drive the crank).
So the TR/gamma curves were correct in shape, but the x-axis labelling did
not correspond to your real crank angle -- it looked like the zero-TR points
landed somewhere odd instead of exactly at your real rest/extreme positions.

FIX
---
Added an explicit, single-source-of-truth frame conversion:

    CRANK_OFFSET_DEG = 90.0   # internal theta value that corresponds to your physical rest
    CRANK_DIRECTION  = -1.0   # -1 = physical angle increases CW; +1 = CCW

    to_internal_theta(user_theta)  -> theta used inside calculate_phi/calculate_gamma/etc.
    to_user_theta(internal_theta)  -> theta you actually read off the crank/motor shaft

Everything that solves the geometry (solve_linkage, refine_extremum) still
works entirely in the INTERNAL frame -- untouched, still correct. The
conversion is applied only at the boundary: when we build the arrays that get
plotted, and when we print th_max/th_min/down_span/up_span, we now report
USER-frame angles (0 deg = rest, CW-positive), which is what you actually
care about on the bench. Nothing about the actual solved geometry (y, dy,
l2, transmission ratios, gamma) changes -- only the angle labelling that
those quantities are reported against.

If your physical setup is actually CCW-positive, or rest isn't at
theta_internal=90, just change CRANK_DIRECTION / CRANK_OFFSET_DEG -- those
two lines are the only thing that defines the user<->internal mapping.
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import os

save_folder = r"/Users/apple/Visual studio code/Flappers/Saves"
os.makedirs(save_folder, exist_ok=True)


# ----------------------------------------------------------------------
# Frame conversion: physical (user) crank angle  <->  internal theta
# ----------------------------------------------------------------------
# Internal convention (used by calculate_phi etc.): rest = 90 deg, CCW+.
# Physical convention you actually measure on the bench: rest = 0 deg, CW+.

CRANK_OFFSET_DEG = 90.0   # internal theta at physical rest position
CRANK_DIRECTION = 1.0     # rotation sense matches internal convention; offset is a pure shift


def to_internal_theta(user_theta_deg):
    """Physical/user crank angle (0=rest, CW+) -> internal theta used by
    calculate_phi/calculate_gamma (90=rest, CCW+)."""
    return (CRANK_OFFSET_DEG + CRANK_DIRECTION * user_theta_deg) % 360.0


def to_user_theta(internal_theta_deg):
    """Internal theta -> physical/user crank angle (0=rest, CW+)."""
    return (CRANK_DIRECTION * (internal_theta_deg - CRANK_OFFSET_DEG)) % 360.0


# ----------------------------------------------------------------------
# Core single-arm (input-arm) kinematics — scale-invariant, internal frame
# ----------------------------------------------------------------------

def get_l2(dx, l1, y, dy):
    """Coupler length consistent with rest position: crank at theta=90 (B=(0,l1)),
    arm-1 tip at (y - dx, dy) (arm 1 horizontal, phi_body = 0)."""
    return np.sqrt((y - dx) ** 2 + (dy - l1) ** 2)


def calculate_phi(dx, l1, theta_deg, y, dy, l2):
    """theta_deg is the INTERNAL angle (90=rest, CCW+).
    Returns phi_body (deg) for a given crank angle theta_deg.
    Returns None if the geometry is not closable (impossible coupler length)."""
    theta = np.radians(theta_deg)
    bx, by = l1 * np.cos(theta), l1 * np.sin(theta)
    ax, ay = -dx, dy
    d = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

    cos_val = (y ** 2 + d ** 2 - l2 ** 2) / (2 * y * d)
    if abs(cos_val) > 1:
        return None
    angle_ba_to_y = np.arccos(np.clip(cos_val, -1, 1))
    angle_ba_to_horiz = np.arctan2(by - ay, bx - ax)
    phi = angle_ba_to_horiz + angle_ba_to_y
    return np.degrees(phi)


def calculate_gamma(dx, l1, theta_deg, y, dy, l2):
    """theta_deg is INTERNAL. Transmission angle (deg) between coupler l2
    and rocker arm y."""
    theta = np.radians(theta_deg)
    bx, by = l1 * np.cos(theta), l1 * np.sin(theta)
    ax, ay = -dx, dy
    d = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    cos_gamma = (l2 ** 2 + y ** 2 - d ** 2) / (2 * l2 * y)
    return np.degrees(np.arccos(np.clip(cos_gamma, -1, 1)))


def phi_array(dx, l1, y, dy, l2, n=720):
    """Internal-frame sweep (theta = 0..360 internal). Used by the solver only."""
    thetas = np.linspace(0, 360, n, endpoint=False)
    raw = [calculate_phi(dx, l1, t, y, dy, l2) for t in thetas]
    vals = np.array([v if v is not None else np.nan for v in raw], dtype=float)
    return thetas, vals


def refine_extremum(dx, l1, y, dy, l2, theta_guess, window=5.0):
    """theta_guess is INTERNAL. Refine a coarse-grid extremum to continuous
    precision using scipy.optimize.minimize_scalar. Returns INTERNAL theta."""
    def f(theta, sign):
        p = calculate_phi(dx, l1, theta, y, dy, l2)
        return sign * p if p is not None else 1e9

    p0 = calculate_phi(dx, l1, theta_guess, y, dy, l2)
    pL = calculate_phi(dx, l1, theta_guess - 1, y, dy, l2)
    pR = calculate_phi(dx, l1, theta_guess + 1, y, dy, l2)
    is_max = (p0 is not None) and (pL is None or p0 >= pL) and (pR is None or p0 >= pR)
    sign = -1 if is_max else 1

    lo, hi = theta_guess - window, theta_guess + window
    res = minimize_scalar(lambda t: f(t, sign), bounds=(lo, hi), method='bounded')
    return res.x, calculate_phi(dx, l1, res.x, y, dy, l2)


# ----------------------------------------------------------------------
# Explicit closability / Grashof-style sanity check
# ----------------------------------------------------------------------

def check_linkage_validity(dx, l1, y, dy, l2, n_check=360):
    """Sweep theta (internal) and confirm the linkage closes everywhere.
    Returns (is_valid, frac_invalid)."""
    thetas = np.linspace(0, 360, n_check, endpoint=False)
    vals = [calculate_phi(dx, l1, t, y, dy, l2) for t in thetas]
    n_bad = sum(v is None for v in vals)
    frac_bad = n_bad / n_check
    return frac_bad == 0.0, frac_bad


# ----------------------------------------------------------------------
# solve_linkage — auto-derived, ratio-based search window (unchanged logic,
# operates entirely in the internal frame, which is correct since it's just
# searching geometry, not reporting an angle to you)
# ----------------------------------------------------------------------

def solve_linkage(dx, l1, target_min_phi_deg, y_range=None, dy_range=None,
                   grid_n=80, theta_n=180, verbose=True):
    if y_range is None:
        y_range = (1.2 * l1, 3.0 * l1)
    if dy_range is None:
        dy_range = (2.0 * dx, 3.6 * dx)

    best_error = float('inf')
    best = None
    for test_y in np.linspace(*y_range, grid_n):
        for test_dy in np.linspace(*dy_range, grid_n):
            l2 = get_l2(dx, l1, test_y, test_dy)
            thetas, phis = phi_array(dx, l1, test_y, test_dy, l2, n=theta_n)
            if np.any(np.isnan(phis)):
                continue
            max_phi, min_phi = phis.max(), phis.min()
            error = abs(max_phi - 0) + abs(min_phi - target_min_phi_deg)
            if error < best_error:
                best_error = error
                best = (test_y, test_dy, l2, thetas[np.argmax(phis)], thetas[np.argmin(phis)])

    if best is None:
        raise ValueError(
            f"No valid (closable) linkage found in search window "
            f"y_range={y_range}, dy_range={dy_range} for dx={dx}, l1={l1}. "
            f"Try widening y_range/dy_range manually."
        )

    test_y, test_dy, l2, _, _ = best
    edge_tol_y = 0.02 * (y_range[1] - y_range[0])
    edge_tol_dy = 0.02 * (dy_range[1] - dy_range[0])
    on_edge = (
        abs(test_y - y_range[0]) < edge_tol_y or abs(test_y - y_range[1]) < edge_tol_y or
        abs(test_dy - dy_range[0]) < edge_tol_dy or abs(test_dy - dy_range[1]) < edge_tol_dy
    )
    if verbose and on_edge:
        print(f"[WARNING] Best fit (y={test_y:.3f}, dy={test_dy:.3f}) lies on the edge "
              f"of the search window y_range={y_range}, dy_range={dy_range}. "
              f"The true optimum may lie outside it -- consider widening manually.")

    is_valid, frac_bad = check_linkage_validity(dx, l1, test_y, test_dy, l2)
    if verbose and not is_valid:
        print(f"[WARNING] Best-fit linkage fails to close for {frac_bad*100:.1f}% "
              f"of the crank cycle. Geometry may be marginal/non-Grashof for this "
              f"target_min_phi_deg -- results in that arc will show as gaps (NaN).")

    return best


# ----------------------------------------------------------------------
# V-rocker extension: rigid output arm at a fixed bend angle from arm 1
# (theta_deg here is INTERNAL -- callers in __main__ convert before calling)
# ----------------------------------------------------------------------

def output_arm_state(dx, l1, y, dy, l2, y2, bend_deg, theta_deg):
    phi_body = calculate_phi(dx, l1, theta_deg, y, dy, l2)
    if phi_body is None:
        return None, None, None

    offset = 180.0 - bend_deg
    phi_output = phi_body + offset

    ax, ay = -dx, dy
    tip_x = ax + y2 * np.cos(np.radians(phi_output))
    tip_y = ay + y2 * np.sin(np.radians(phi_output))
    return phi_body, phi_output, (tip_x, tip_y)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # ============================================================
    # INPUTS — replace these for YOUR airframe.
    # ============================================================
    dx_val = 12.5
    l1_val = 10.0
    target_down = -50.0
    y2_val = 20.0
    bend_deg = 165.0

    motor_rpm = 43000
    gearbox_ratio = 35.94

    y_range_override = None
    dy_range_override = None

    # 1) Solve the input-arm geometry (internal frame -- unaffected by the
    #    user-frame fix, since this is pure geometry, not a reported angle)
    y, dy, l2, th_max_g, th_min_g = solve_linkage(
        dx_val, l1_val, target_down,
        y_range=y_range_override, dy_range=dy_range_override,
    )

    # 2) Refine the true extrema theta/phi to continuous precision (internal frame)
    th_max_int, p_max = refine_extremum(dx_val, l1_val, y, dy, l2, th_max_g)
    th_min_int, p_min = refine_extremum(dx_val, l1_val, y, dy, l2, th_min_g)

    # Convert extrema crank angles to YOUR physical frame (0=rest, CW+) for reporting
    th_max = to_user_theta(th_max_int)
    th_min = to_user_theta(th_min_int)

    # Stroke spans are differences of physical angles taken the same direction
    # the crank actually turns (CW, i.e. increasing user_theta)
    down_span = (th_min - th_max) % 360
    up_span = 360 - down_span

    # 3) Full-cycle sweep, built directly in the USER (physical) frame so
    #    every array below is indexed by the crank angle you actually read
    #    off the shaft, with rest at 0 deg and increasing CW.
    user_thetas = np.linspace(0, 360, 720, endpoint=False)
    phi_body_list, phi_out_list, tipx_list, tipy_list = [], [], [], []
    for ut in user_thetas:
        it = to_internal_theta(ut)
        pb, po, tip = output_arm_state(dx_val, l1_val, y, dy, l2, y2_val, bend_deg, it)
        phi_body_list.append(pb)
        phi_out_list.append(po)
        tipx_list.append(tip[0] if tip else np.nan)
        tipy_list.append(tip[1] if tip else np.nan)
    phi_out_arr = np.array(phi_out_list, dtype=float)
    tipx_arr = np.array(tipx_list, dtype=float)

    # 4) Gearbox / frequency analysis (unaffected by frame convention)
    crank_rpm = motor_rpm / gearbox_ratio
    flapping_freq_hz = crank_rpm / 60.0
    total_sweep = abs(p_max - p_min)
    avg_tr_down = total_sweep / down_span
    avg_tr_up = total_sweep / up_span
    avg_tr_overall = total_sweep / 180.0
    total_reduction_down = gearbox_ratio / avg_tr_down
    total_reduction_up = gearbox_ratio / avg_tr_up

    print("--- Validated Input-Arm (Rocker) Geometry ---")
    print(f"dx (input):            {dx_val:.3f} mm")
    print(f"l1 (input):            {l1_val:.3f} mm")
    print(f"Arm-1 length (y):      {y:.3f} mm")
    print(f"Pivot height (dy):     {dy:.3f} mm")
    print(f"Coupler length (l2):   {l2:.3f} mm")
    print(f"phi_body at rest (user_theta=0, should be 0): "
          f"{calculate_phi(dx_val, l1_val, to_internal_theta(0), y, dy, l2):.4f} deg")
    print("-" * 55)
    print("Angles below are in YOUR physical crank frame: 0 deg = rest, "
          "increasing CLOCKWISE.")
    print(f"Max phi (arm 1):  {p_max:.4f} deg at crank angle = {th_max:.3f} deg")
    print(f"Min phi (arm 1):  {p_min:.4f} deg at crank angle = {th_min:.3f} deg")
    print(f"Total sweep (arm 1): {total_sweep:.4f} deg")
    print(f"Down-stroke crank span: {down_span:.3f} deg")
    print(f"Up-stroke crank span:   {up_span:.3f} deg")
    print(f"Asymmetry vs flat 180/180 split: {abs(down_span - up_span):.3f} deg")
    print("-" * 55)
    print(f"V-Rocker bend angle (interior): {bend_deg} deg  (offset = {180 - bend_deg:.1f} deg)")
    print(f"Output arm length (y2): {y2_val} mm")
    print(f"Output arm phi range: {np.nanmin(phi_out_arr):.3f} to {np.nanmax(phi_out_arr):.3f} deg")
    print(f"Output-arm tip x-travel: {np.nanmax(tipx_arr) - np.nanmin(tipx_arr):.3f} mm")
    print("-" * 55)
    print("--- Gearbox & Transmission Analysis (asymmetry-aware) ---")
    print(f"Motor Speed:            {motor_rpm} RPM")
    print(f"Gearbox Ratio:          {gearbox_ratio}:1")
    print(f"Crank Speed:            {crank_rpm:.2f} RPM")
    print(f"Flapping Frequency:     {flapping_freq_hz:.2f} Hz")
    print(f"Avg TR (down-stroke):   {avg_tr_down:.4f} wing-deg/crank-deg")
    print(f"Avg TR (up-stroke):     {avg_tr_up:.4f} wing-deg/crank-deg")
    print(f"Avg TR (flat /180, ref only): {avg_tr_overall:.4f} wing-deg/crank-deg")
    print(f"Total reduction, down-stroke: {total_reduction_down:.2f}:1")
    print(f"Total reduction, up-stroke:   {total_reduction_up:.2f}:1")

    # ------------------------------------------------------------------
    # PLOTTING — built entirely in the USER (physical, CW, rest=0) frame
    # ------------------------------------------------------------------
    theta_plot = user_thetas  # already 0..360, physical, CW, rest=0
    internal_plot = np.array([to_internal_theta(ut) for ut in theta_plot])
    phi_plot = np.array([calculate_phi(dx_val, l1_val, it, y, dy, l2) for it in internal_plot])
    gamma_plot = np.array([calculate_gamma(dx_val, l1_val, it, y, dy, l2) for it in internal_plot])

    # Differentiate phi w.r.t. the USER theta (not internal) -- this is what
    # makes TR fall to exactly zero at the extrema as plotted on this axis.
    # Direction reversal (CRANK_DIRECTION=-1) just flips the sign, which
    # np.abs() below removes, so the magnitude/shape is identical either way.
    phi_rad = np.radians(phi_plot)
    theta_rad = np.radians(theta_plot)
    tr_instant = np.abs(np.gradient(phi_rad, theta_rad))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(theta_plot, tr_instant, color='blue', lw=2)
    ax1.fill_between(theta_plot, tr_instant, alpha=0.2, color='blue')
    ax1.set_ylabel("Instantaneous TR\n($\\omega_{out} / \\omega_{in}$)")
    ax1.set_title("Mechanism Transmission Characteristics\n(crank angle: 0°=rest, CW+)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(theta_plot, gamma_plot, color='red', lw=2)
    ax2.axhline(90, color='black', linestyle=':', label='Ideal (90°)')
    ax2.axhline(40, color='orange', linestyle='--', label='Min Recommended (40°)')
    ax2.set_ylabel("Transmission Angle (γ)\n(Quality in Degrees)")
    ax2.set_xlabel("Crank Angle (Degrees, 0°=rest, CW+)")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    out1 = os.path.join(save_folder, 'mechanism_transmission_characteristics.png')
    plt.savefig(out1, dpi=180, bbox_inches='tight')
    print(f'Saved → {out1}')
    plt.show()

    # ------------------------------------------------------------------
    # EXTRA PLOTS: sin(gamma), TR x gamma, slope of TR x gamma
    # ------------------------------------------------------------------
    sin_gamma = np.sin(np.radians(gamma_plot))
    tr_times_gamma = np.abs(tr_instant * gamma_plot)
    slope_tr_gamma = np.abs(np.gradient(tr_times_gamma, theta_plot))

    mean_val = np.mean(tr_times_gamma)
    max_slope_idx = np.argmax(slope_tr_gamma)
    max_slope = slope_tr_gamma[max_slope_idx]
    max_slope_theta = theta_plot[max_slope_idx]
    avg_abs_slope = np.mean(slope_tr_gamma)

    print("-" * 55)
    print("--- TR x gamma Curve Analysis (crank angle: 0°=rest, CW+) ---")
    print(f"Mean value of (TR x gamma):              {mean_val:.4f} deg")
    print(f"Max |slope| of (TR x gamma):             {max_slope:.4f} deg/deg, "
          f"at crank angle = {max_slope_theta:.2f} deg")
    print(f"Average |slope| (magnitude) over cycle:  {avg_abs_slope:.4f} deg/deg")
    print("-" * 55)

    fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    ax3.plot(theta_plot, sin_gamma, color='green', lw=2)
    ax3.fill_between(theta_plot, sin_gamma, alpha=0.2, color='green')
    ax3.axhline(1.0, color='black', linestyle=':', label='Ideal (γ=90°, sin=1)')
    ax3.set_ylabel("sin(γ)\n(Force Transmission\nEfficiency Factor)")
    ax3.set_title("Transmission Angle Quality & Combined TR Metric\n(crank angle: 0°=rest, CW+)")
    ax3.legend(loc='lower right')
    ax3.grid(True, linestyle='--', alpha=0.7)

    ax4.plot(theta_plot, tr_times_gamma, color='purple', lw=2)
    ax4.fill_between(theta_plot, tr_times_gamma, alpha=0.2, color='purple')
    ax4.axhline(mean_val, color='black', linestyle='--', label=f'Mean ({mean_val:.2f}°)')
    ax4.set_ylabel("TR × γ\n(deg)")
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.7)

    ax5.plot(theta_plot, slope_tr_gamma, color='darkorange', lw=2, label='|Slope of TR × γ|')
    ax5.axhline(avg_abs_slope, color='black', linestyle='--',
                label=f'Avg |slope| ({avg_abs_slope:.4f} deg/deg)')
    ax5.axhline(max_slope, color='red', linestyle=':',
                label=f'Max |slope| ({max_slope:.4f} deg/deg)')
    ax5.set_ylim(bottom=0)
    ax5.set_xlabel("Crank Angle (Degrees, 0°=rest, CW+)")
    ax5.set_ylabel("|d(TR × γ)/dθ|\n(deg/deg)")
    ax5.legend(loc='upper right')
    ax5.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    out2 = os.path.join(save_folder, 'tr_gamma_analysis.png')
    plt.savefig(out2, dpi=180, bbox_inches='tight')
    print(f'Saved → {out2}')
    plt.show()
