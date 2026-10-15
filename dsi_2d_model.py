"""
Cross-sectional model of direct steam injection (DSI) into a milk pipe.

Resolves the temperature across the pipe in horizontal layers, marching along
the pipe from the injector. Milk is treated as water. Combines a choked-nozzle
limit, an integral jet model, a line-pressure saturation cap, turbulent and
jet-enhanced lateral mixing (solved implicitly) and wall heat loss. Used to
compare injector size, angle, position and pipe insulation.

Equations, assumptions and sources are documented in the final report.

Run:  python dsi_2d_model_v6.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass

_trapz = getattr(np, "trapezoid", None) or np.trapz

# ============================================================================
# INPUTS
# ============================================================================
milk_velocity = 2.0          # m/s  bulk mean
milk_T_in     = 6.0          # degC
cp            = 4.186        # kJ/kg.K
rho           = 1000.0       # kg/m3
mu            = 1.47e-3      # Pa.s
k_therm       = 0.573        # W/m.K

T_target      = 72.0         # degC  (pasteurisation)

# Steam supply (saturated, 6 bar abs)
steam_pressure = 6.0e5       # Pa abs
h_g            = 2756.1      # kJ/kg
h_fg           = 2085.6      # kJ/kg  (IAPWS-IF97 at 6 bar(a))
rho_g          = 1.0/0.3156  # kg/m3
k_isen         = 1.135
Cd             = 0.85        # discharge coefficient of a plain steam orifice

# Milk line
line_pressure  = 3.0e5       # Pa abs -- sets the saturation cap and flash check

# Geometry
pipe_length     = 41.0       # m  (sized so the FASTEST element gets a 15 s hold; see report)
pipe_diameter   = 0.05       # m
inj_position    = 0.2        # m from inlet
nozzle_diameter = 0.005      # m  (basis nozzle)

# Wall / heat loss
T_amb          = 10.0        # degC
wall_thickness = 1.6e-3      # m  stainless
k_wall         = 16.0        # W/m.K
ins_thickness  = 0.05        # m  generic pipe insulation
k_ins          = 0.04        # W/m.K (typical mineral fibre / foam)
h_out          = 10.0        # W/m2.K  natural convection + radiation

# Jet model constants (heuristic, see report)
alpha_ent = 0.08
beta_ent  = 0.6
C_jet     = 0.05
L_mix_D   = 5.0
DEPOSIT_MODE = "entrainment"   # "entrainment" | "excess_temp" (v5 heuristic)

GEOMETRY = "pipe"
PROFILE  = "power17"
n_power  = 7.0
dx_target = 1e-3             # m  axial step (jet deposition happens within ~20 mm, keep <= 2 mm)
ny        = 40
nx        = int(round(pipe_length/dx_target)) + 1

# ============================================================================
# CASE DEFINITION
# ============================================================================
@dataclass
class Case:
    label:     str
    nozzle:    str   = "required"   # "basis" (5 mm, choked) | "required" (hits T_target)
    angle:     float = 45.0         # deg from pipe axis, 90 = radial
    position:  str   = "top"        # "top" | "bottom" | "centre" | "ring"
    wall:      str   = "adiabatic"  # "adiabatic" | "bare" | "insulated"
    n_nozzles: int   = 1            # >1 only meaningful for "ring"

# ============================================================================
# STEAM TABLES (saturation temperature, Antoine 99-374 degC)
# ============================================================================
def T_sat_of_P(P_pa):
    P_mmHg = P_pa/133.322
    return 1810.94/(8.14019 - np.log10(P_mmHg)) - 244.485

T_sat_steam = T_sat_of_P(steam_pressure)
T_sat_line  = T_sat_of_P(line_pressure)
flash_margin = T_sat_line - T_target

# ============================================================================
# GEOMETRY AND VELOCITY PROFILE
# ============================================================================
R    = pipe_diameter/2.0
area = np.pi*R**2
milk_mdot = rho*area*milk_velocity
dx = pipe_length/(nx-1)
x_axis = np.linspace(0.0, pipe_length, nx)

dy     = pipe_diameter/ny
y      = (np.arange(ny) + 0.5)*dy
y_face = np.arange(1, ny)*dy
y_edge = np.arange(ny+1)*dy

def chord(yy):
    s = np.clip(yy - R, -R, R)
    return 2.0*np.sqrt(np.maximum(R**2 - s**2, 0.0))

if GEOMETRY == "pipe":
    A_cell = np.array([_trapz(chord(np.linspace(y_edge[j], y_edge[j+1], 51)),
                              np.linspace(y_edge[j], y_edge[j+1], 51))
                       for j in range(ny)])
    A_cell *= area/A_cell.sum()
    L_iface = chord(y_face)
    phi_edge = np.arcsin(np.clip((y_edge - R)/R, -1, 1))
    P_cell   = 2.0*R*np.diff(phi_edge)          # wetted arc per strip, sum = pi*D
else:
    w = area/pipe_diameter
    A_cell  = np.full(ny, w*dy)
    L_iface = np.full(ny-1, w)
    P_cell  = np.zeros(ny); P_cell[0] = P_cell[-1] = np.pi*pipe_diameter/2

if PROFILE == "power17":
    y_wall = np.minimum(y, pipe_diameter - y)
    u_prof = np.maximum(y_wall/R, 1e-6)**(1.0/n_power)
else:
    u_prof = np.ones(ny)
u_prof *= milk_mdot/(rho*(u_prof*A_cell).sum())
mdot_cell = rho*u_prof*A_cell
assert abs(mdot_cell.sum() - milk_mdot) < 1e-9

# ============================================================================
# TURBULENCE AND WALL HEAT TRANSFER
# ============================================================================
nu    = mu/rho
alpha = k_therm/(rho*cp*1000.0)
Pr    = mu*cp*1000.0/k_therm
Re    = milk_velocity*pipe_diameter/nu
f     = (0.790*np.log(Re) - 1.64)**-2
ustar = milk_velocity*np.sqrt(f/8.0)
D_pipe = alpha + 0.07*R*ustar/0.85
y_plus_1 = (dy/2)*ustar/nu

Nu_i = 0.023*Re**0.8*Pr**0.4
h_in = Nu_i*k_therm/pipe_diameter                       # W/m2.K

def wall_U(kind):
    """Overall coefficient W/m2.K referenced to the inner surface."""
    if kind == "adiabatic":
        return 0.0
    r_i, r_o = R, R + wall_thickness
    Rth = 1.0/h_in + r_i*np.log(r_o/r_i)/k_wall
    if kind == "insulated":
        r_x  = r_o + ins_thickness
        Rth += r_i*np.log(r_x/r_o)/k_ins + r_i/(r_x*h_out)
    else:
        Rth += r_i/(r_o*h_out)
    return 1.0/Rth

# ============================================================================
# NOZZLE
# ============================================================================
def choked_nozzle(P0, rho0, k, A, Cd=1.0):
    r        = 2.0/(k+1.0)
    P_star   = P0*r**(k/(k-1.0))
    rho_star = rho0*r**(1.0/(k-1.0))
    a_star   = np.sqrt(k*P_star/rho_star)
    return dict(P_star=P_star, rho_star=rho_star, a_star=a_star,
                G=rho_star*a_star, mdot_max=Cd*rho_star*a_star*A)

def steam_for_target(T_out):
    return milk_mdot*cp*(T_out - milk_T_in)/(h_g - cp*T_out)

noz_ref  = choked_nozzle(steam_pressure, rho_g, k_isen, 1.0, Cd)   # per m2
G_choked = Cd*noz_ref['G']
mdot_req = steam_for_target(T_target)
mdot_basis_single = G_choked*np.pi*(nozzle_diameter/2)**2
d_required_single = np.sqrt(4.0*mdot_req/(G_choked*np.pi))

def nozzle_for(case):
    """Return (total steam kg/s, per-nozzle diameter, number of nozzles)."""
    n = case.n_nozzles if case.position == "ring" else 1
    if case.nozzle == "basis":
        return n*mdot_basis_single, nozzle_diameter, n
    return mdot_req, np.sqrt(4.0*(mdot_req/n)/(G_choked*np.pi)), n

# ============================================================================
# JET IN CROSSFLOW
# ============================================================================
def jet_trajectory(mdot_j, d_noz, y0, dir_sign, angle_deg, m_cap):
    """Integral jet model. Returns dict of path arrays and metadata.

    Terminates when: the jet velocity decays to the crossflow ('decayed'), it
    hits a wall ('wall'), it leaves the pipe ('exit'), or it has entrained the
    whole available pipe flow / grown to the pipe radius ('confined' -- the
    jet has become the pipe flow, i.e. a confined-jet mixer)."""
    U      = milk_velocity
    v_j    = noz_ref['a_star']
    rho_j  = noz_ref['rho_star']
    theta  = np.radians(angle_deg)
    d_eq   = d_noz*np.sqrt(rho_j/rho)

    m  = mdot_j
    Mx = m*v_j*np.cos(theta)
    My = dir_sign*m*v_j*np.sin(theta)
    xj, yj = inj_position, y0
    ds = 2.5e-4
    xs, ys, bs, ms, vs = [], [], [], [], []
    end = "decayed"
    for _ in range(200000):
        vx, vy = Mx/m, My/m
        v = np.hypot(vx, vy)
        b = np.sqrt(m/(rho*np.pi*v))
        xs.append(xj); ys.append(yj); bs.append(b); ms.append(m); vs.append(v)
        if v <= 1.05*U:
            end = "decayed"; break
        if xj >= pipe_length:
            end = "exit"; break
        if m >= m_cap or b >= R:
            end = "confined"; break
        phi  = np.arctan2(abs(vy), vx)
        dmds = rho*2*np.pi*b*(alpha_ent*abs(v - U*np.cos(phi)) + beta_ent*U*abs(np.sin(phi)))
        dm   = dmds*ds
        m   += dm; Mx += U*dm
        xj  += vx/v*ds; yj += vy/v*ds
        if yj <= 0.0 or yj >= pipe_diameter:
            yj = np.clip(yj, 0.0, pipe_diameter)
            end = "wall"
            xs.append(xj); ys.append(yj); bs.append(b); ms.append(m); vs.append(v)
            break
    xs, ys, bs, ms, vs = map(np.array, (xs, ys, bs, ms, vs))
    J    = rho_j*v_j**2/(rho*U**2)
    B    = cp*(T_sat_steam - milk_T_in)/h_fg
    G    = mdot_j/(np.pi*(d_noz/2)**2)
    L_c  = 0.2588/B*np.sqrt(G/275.0)*d_noz            # Kerney, along the jet
    pen  = np.abs(ys - y0).max()
    return dict(x=xs, y=ys, b=bs, m=ms, v=vs, ds=ds, end=end, d_eq=d_eq, J=J,
                L_c=L_c, r_d=np.sqrt(J)*d_eq, pen=pen,
                x_pen=xs[np.argmax(np.abs(ys - y0))], M_jet=mdot_j*v_j,
                v_end=vs[-1], m_ratio=ms[-1]/ms[0])

def deposit(S, traj, frac):
    """Add a jet's heat-deposition weights (summing to frac) to the source S.

    DEPOSIT_MODE = "entrainment" (default): fluid entrained at station s is
    heated to the jet's local excess temperature, dT(s) = dT0*m0/m(s), so the
    heat handed over per step is  m0*dm/m^2. Integrated along the path this
    gives the fraction 1 - m0/m_end; the remainder m0/m_end is still inside
    the jet core at termination and is deposited there -- over the local jet
    width b_end, or uniformly across the section if the jet became confined.

    DEPOSIT_MODE = "excess_temp": the v5 heuristic, rate ~ 1/m along the path
    (kept for comparison).
    """
    xs, ys, bs, ms = traj['x'], traj['y'], traj['b'], traj['m']
    m0, m_end = ms[0], ms[-1]
    if DEPOSIT_MODE == "entrainment":
        dm  = np.diff(ms)
        wk  = m0*dm/ms[:-1]**2
        wk  = np.append(wk, 0.0)
        rem = m0/m_end
    else:
        wk  = traj['ds']/ms
        wk /= wk.sum()
        rem = 0.0
    wk *= frac
    for xk, yk, bk, w in zip(xs, ys, bs, wk):
        if w <= 0.0:
            continue
        i = int(np.clip(round(xk/dx), 0, nx-1))
        half = max(bk, dy/2)
        mask = np.abs(y - yk) <= half
        if not mask.any():
            mask = np.abs(y - yk) == np.abs(y - yk).min()
        S[i, mask] += w/mask.sum()
    if rem > 0.0:
        i = int(np.clip(round(xs[-1]/dx), 0, nx-1))
        if traj['end'] == "confined":
            S[i] += frac*rem*mdot_cell/mdot_cell.sum()       # uniform dT
        else:
            mask = np.abs(y - ys[-1]) <= max(bs[-1], dy/2)
            S[i, mask] += frac*rem/mask.sum()

def build_source(case):
    mdot_s, d_noz, n = nozzle_for(case)
    S = np.zeros((nx, ny))
    if case.position == "top":
        jets = [(pipe_diameter, -1.0, 1.0)]
    elif case.position == "bottom":
        jets = [(0.0, +1.0, 1.0)]
    elif case.position == "centre":
        jets = [(R, -1.0, 1.0)]
    elif case.position == "ring":
        jets = [(pipe_diameter, -1.0, 0.5), (0.0, +1.0, 0.5)]
    else:
        raise ValueError(case.position)
    trajs = []
    for y0, sgn, frac in jets:
        t = jet_trajectory(mdot_s*frac, d_noz, y0, sgn, case.angle,
                           m_cap=frac*milk_mdot + mdot_s*frac)
        deposit(S, t, frac)
        trajs.append(t)
    S /= S.sum()
    return S, trajs, mdot_s, d_noz, n

# ============================================================================
# SOLVER
# ============================================================================
def allocate_condensation(mdot_s, w, m_L, H_L, T_cap):
    cap = np.maximum((m_L*cp*T_cap - H_L)/(h_g - cp*T_cap), 0.0)
    dm  = np.zeros(len(w)); rem = mdot_s; ww = w.copy()
    for _ in range(100):
        tot = ww.sum()
        if rem <= 1e-14 or tot <= 0.0:
            break
        take = np.minimum(rem*ww/tot, cap - dm)
        dm  += take; rem -= take.sum()
        ww   = np.where(cap - dm > 1e-15, ww, 0.0)
    return dm, rem

def thomas_diffuse(T, m, kf):
    """Solve  m_j (T'_j - T_j) = kf_{j-1}(T'_{j-1}-T'_j) + kf_j (T'_{j+1}-T'_j)  for T'."""
    n = len(T)
    a = np.zeros(n); b = m.copy(); c = np.zeros(n)
    b[:-1] += kf; b[1:] += kf
    a[1:] = -kf; c[:-1] = -kf
    d = m*T
    # forward sweep
    cp_ = np.zeros(n); dp = np.zeros(n)
    cp_[0] = c[0]/b[0]; dp[0] = d[0]/b[0]
    for j in range(1, n):
        den = b[j] - a[j]*cp_[j-1]
        cp_[j] = c[j]/den
        dp[j] = (d[j] - a[j]*dp[j-1])/den
    x = np.zeros(n); x[-1] = dp[-1]
    for j in range(n-2, -1, -1):
        x[j] = dp[j] - cp_[j]*x[j+1]
    return x

def solve(case):
    S, trajs, mdot_s, d_noz, n = build_source(case)
    U_w = wall_U(case.wall)

    # jet-enhanced lateral diffusivity
    M_jet = sum(t['M_jet'] for t in trajs)
    U_j   = np.sqrt(M_jet/(rho*area))
    D_x   = D_pipe + np.where(x_axis >= inj_position,
                              C_jet*U_j*pipe_diameter*np.exp(-(x_axis - inj_position)/(L_mix_D*pipe_diameter)),
                              0.0)

    m_L = mdot_cell.copy()
    H_L = m_L*cp*milk_T_in
    T_field = np.zeros((nx, ny)); m_field = np.zeros((nx, ny))
    T_field[0], m_field[0] = H_L/(m_L*cp), m_L

    carry, last_w = 0.0, S[S.sum(axis=1).argmax()]
    n_sat_max, n_clip, Q_loss, uncondensed = 0, 0, 0.0, 0.0   # n_clip retained for API, always 0
    for i in range(1, nx):
        s_i  = S[i].sum()
        todo = mdot_s*s_i + carry
        if todo > 1e-14:
            w = S[i] if s_i > 0 else last_w
            dm, carry = allocate_condensation(todo, w, m_L, H_L, T_sat_line)
            m_L += dm; H_L += dm*h_g
            if s_i > 0: last_w = S[i]
            n_sat_max = max(n_sat_max, int((H_L/(m_L*cp) > T_sat_line - 1e-6).sum()))

        T = H_L/(m_L*cp)
        if U_w > 0.0:
            q_w = U_w*P_cell*dx*(T - T_amb)/1000.0        # kW per strip
            H_L -= q_w; Q_loss += q_w.sum()
            T = H_L/(m_L*cp)

        # Lateral turbulent diffusion, IMPLICIT (backward Euler in x).
        # Unconditionally stable, so the large jet-region diffusivity is
        # handled correctly and results are grid-independent. Conductances
        # are symmetric, so energy is conserved to round-off.
        kf = rho*D_x[i]*L_iface*dx/dy                     # kg/s per interface
        T  = thomas_diffuse(T, m_L, kf)
        H_L = m_L*cp*T
        T_field[i], m_field[i] = H_L/(m_L*cp), m_L
    uncondensed = carry

    T_out = H_L.sum()/(m_L.sum()*cp)
    # Residence times downstream of the injector, using the OUTLET velocity
    # (condensate adds ~11% to the mass flow and hence to the velocity).
    u_scale = m_L.sum()/milk_mdot
    L_hold  = pipe_length - inj_position
    t_mean  = L_hold/(milk_velocity*u_scale)
    t_fast  = L_hold/(u_prof.max()*u_scale)       # strip-model centreline (u_max/u_mean ~ 1.10)
    t_fast_pl = L_hold/(1.224*milk_velocity*u_scale)  # true 1/7-power centreline, u_max/u_mean = 1.224
    t_wall  = L_hold/(u_prof.min()*u_scale)
    Tb = (T_field*m_field).sum(axis=1)/m_field.sum(axis=1)
    spread = T_field.max(axis=1) - T_field.min(axis=1)
    ix = int(round(inj_position/dx))
    after = np.where(spread[ix:] < 1.0)[0]
    L_mix = (after[0]*dx) if len(after) else np.nan
    top, bot = T_field[:, -1], T_field[:, 0]
    # Distance downstream of the injector over which each wall strip stays
    # more than 0.1 degC hotter than the product (mixing-cup) temperature.
    hot_top = np.where(top[ix:] - Tb[ix:] > 0.1)[0]
    hot_bot = np.where(bot[ix:] - Tb[ix:] > 0.1)[0]
    hot_top = (hot_top[-1] + 1)*dx if len(hot_top) else 0.0
    hot_bot = (hot_bot[-1] + 1)*dx if len(hot_bot) else 0.0
    return dict(case=case, T=T_field, m=m_field, Tb=Tb, T_out=T_out, spread=spread,
                L_mix=L_mix, top=top, bot=bot, mdot_s=mdot_s, d_noz=d_noz, n=n,
                trajs=trajs, S=S, U_w=U_w, Q_loss=Q_loss, n_sat=n_sat_max,
                n_clip=n_clip, uncondensed=uncondensed, D_x=D_x,
                over70_top=(top > 70).sum()*dx, over70_bot=(bot > 70).sum()*dx,
                hot_top=hot_top, hot_bot=hot_bot,
                t_mean=t_mean, t_fast=t_fast, t_fast_pl=t_fast_pl, t_wall=t_wall, T_min_out=T_field[-1].min())

# ============================================================================
# REPORTING
# ============================================================================
def header():
    print("="*78)
    print(f"DSI 2D model v6 | GEOMETRY={GEOMETRY} PROFILE={PROFILE} nx={nx} ny={ny}")
    print("="*78)
    print(f"  milk {milk_mdot:.2f} kg/s at {milk_velocity} m/s, Re={Re:,.0f}, Pr={Pr:.1f}")
    print(f"  u*={ustar:.4f} m/s  D_pipe={D_pipe:.2e} m2/s  first cell y+={y_plus_1:.0f}")
    print(f"  steam supply {steam_pressure/1e5:.1f} bar (T_sat {T_sat_steam:.1f} degC)")
    print(f"  milk line    {line_pressure/1e5:.1f} bar (T_sat {T_sat_line:.1f} degC) -> cap on cell T")
    print(f"  flash check: target {T_target:.0f} degC is {flash_margin:.1f} degC below line T_sat"
          + ("  *** FLASH RISK ***" if flash_margin < 5 else "  OK"))
    print(f"  steam required for {T_target:.0f} degC: {mdot_req*3600:,.0f} kg/h")
    print(f"  basis nozzle {nozzle_diameter*1000:.1f} mm passes {mdot_basis_single*3600:,.0f} kg/h "
          f"(choked, Cd={Cd}) -> single required nozzle {d_required_single*1000:.1f} mm")
    print(f"  wall U (W/m2K, inner area): bare {wall_U('bare'):.1f}, insulated {wall_U('insulated'):.2f}"
          f"  (h_in={h_in:,.0f})")

def report(res):
    c, t = res['case'], res['trajs'][0]
    print("\n" + "-"*78)
    print(f"{c.label}")
    print(f"  nozzle {c.nozzle}: {res['n']} x {res['d_noz']*1000:.1f} mm, {res['mdot_s']*3600:,.0f} kg/h | "
          f"angle {c.angle:.0f} deg | {c.position} | wall {c.wall}")
    print(f"  jet: J={t['J']:.0f}, r*d_eq={t['r_d']*1000:.1f} mm, Kerney L_c={t['L_c']*1000:.1f} mm, "
          f"penetration {t['pen']/pipe_diameter*100:.0f}% D at x={t['x_pen']*1000-inj_position*1000:.0f} mm "
          f"downstream, ends by '{t['end']}' at x={t['x'][-1]*1000-inj_position*1000:.0f} mm "
          f"(m/m0={t['m_ratio']:.1f}, {100*(1-1/t['m_ratio']):.0f}% of heat deposited along path)")
    print(f"  outlet {res['T_out']:.2f} degC, outlet spread {res['spread'][-1]:.2f} degC, "
          + (f"mixing length (spread<1 degC) {res['L_mix']*1000:.0f} mm" if np.isfinite(res['L_mix'])
             else "mixing length (spread<1 degC) > pipe length"))
    print(f"  peak strip T: top {res['top'].max():.1f}, bottom {res['bot'].max():.1f} degC | "
          f"cells at T_sat(line): {res['n_sat']} of {ny} | top>70 for {res['over70_top']*1000:.0f} mm, "
          f"bottom>70 for {res['over70_bot']*1000:.0f} mm")
    print(f"  wall hotter than product (>0.1 degC) downstream of injector: "
          f"top {res['hot_top']*1000:.0f} mm, bottom {res['hot_bot']*1000:.0f} mm")
    if res['uncondensed'] > 1e-9:
        print(f"  *** {res['uncondensed']/res['mdot_s']*100:.1f}% of steam left the pipe uncondensed ***")
    print(f"  hold downstream of injector: mean {res['t_mean']:.1f} s, fastest element {res['t_fast']:.1f} s, "
          f"(1/7-law centreline {res['t_fast_pl']:.1f} s), wall layer {res['t_wall']:.1f} s | coldest layer at outlet {res['T_min_out']:.2f} degC")
    if c.wall != "adiabatic":
        print(f"  wall heat loss {res['Q_loss']*1000:.0f} W "
              f"({res['Q_loss']/(res['mdot_s']*h_g)*100:.2f}% of steam duty), outlet {res['T_out']:.2f} degC")

# ============================================================================
# PLOTS
# ============================================================================
def plot_sweep(results, fname, title):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.8*n), squeeze=False)
    for row, res in enumerate(results):
        ax1, ax2 = axes[row]
        vmax = min(res['T_out'] + 25.0, T_sat_line)
        pcm = ax1.pcolormesh(x_axis, y*1000, res['T'].T, shading="auto", cmap="inferno",
                             vmin=milk_T_in, vmax=vmax)
        for t in res['trajs']:
            ax1.plot(t['x'], t['y']*1000, color="cyan", lw=1, ls="--")
        ax1.set_xlim(max(0, inj_position - 0.05), inj_position + 0.4); ax1.set_ylim(0, pipe_diameter*1000)
        ax1.set_xlabel("x (m)"); ax1.set_ylabel("pipe height (mm)")
        ax1.set_title(f"{res['case'].label}\nT field (capped {vmax:.0f} degC), jet path dashed", fontsize=9)
        fig.colorbar(pcm, ax=ax1, label="T (degC)", extend="max")

        ax2.plot(x_axis, res['Tb'], lw=2, color="crimson", label="mixing-cup")
        ax2.plot(x_axis, res['top'], lw=1.2, color="darkorange", label="top strip")
        ax2.plot(x_axis, res['bot'], lw=1.2, color="steelblue", label="bottom strip")
        ax2.plot(x_axis, res['T'][:, ny//2], lw=1.0, color="grey", label="centre strip")
        ax2.axhline(T_target, color="k", ls=":", lw=1)
        ax2.axvline(inj_position, color="cyan", ls="--", lw=1)
        ax2.set_xlim(max(0, inj_position - 0.05), inj_position + 0.8)
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("T (degC)")
        ax2.set_title(f"outlet {res['T_out']:.2f} degC, spread {res['spread'][-1]:.2f} degC, "
                      f"L_mix {res['L_mix']*1000:.0f} mm" if np.isfinite(res['L_mix']) else f"L_mix > pipe length",
                      fontsize=9)
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  saved {fname}")

def plot_summary(all_results, fname):
    labels = [r['case'].label for r in all_results]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    idx = np.arange(len(labels))
    axes[0].barh(idx, [r['spread'][-1] for r in all_results], color="crimson")
    axes[0].set_xlabel("outlet spread, max-min (degC)")
    axes[1].barh(idx, [max(r['top'].max(), r['bot'].max()) for r in all_results], color="darkorange")
    axes[1].set_xlabel("peak wall-strip T (degC)")
    axes[2].barh(idx, [r['L_mix']*1000 if np.isfinite(r['L_mix']) else pipe_length*1000
                       for r in all_results], color="steelblue")
    axes[2].set_xlabel("mixing length to <1 degC spread (mm)")
    for ax in axes:
        ax.set_yticks(idx); ax.set_yticklabels(labels, fontsize=8); ax.grid(alpha=0.3, axis="x")
        ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close(fig)
    print(f"  saved {fname}")

# ============================================================================
# SENSITIVITY SWEEPS
# ============================================================================
if __name__ == "__main__":
    header()

    sweeps = {
        "1_nozzle": ("Nozzle size (top, 45 deg, adiabatic)", [
            Case("basis 5 mm nozzle (choked)",  nozzle="basis"),
            Case("required nozzle for 72 degC", nozzle="required"),
        ]),
        "2_angle": ("Injection angle (required nozzle, top, adiabatic)", [
            Case("90 deg (radial)",    angle=90.0),
            Case("45 deg",             angle=45.0),
            Case("20 deg (near-axial)", angle=20.0),
        ]),
        "3_position": ("Injector position (required nozzle, adiabatic)", [
            Case("top wall, 45 deg",             position="top"),
            Case("bottom wall, 45 deg",          position="bottom"),
            Case("centre lance, co-axial (0 deg)", position="centre", angle=0.0),
            Case("ring, 4 nozzles, 45 deg",      position="ring", n_nozzles=4),
        ]),
        "4_wall": ("Wall heat loss, ambient 10 degC (required nozzle, top, 45 deg)", [
            Case("adiabatic",              wall="adiabatic"),
            Case("bare stainless",         wall="bare"),
            Case("50 mm insulation (k = 0.04 W/m.K)", wall="insulated"),
        ]),
    }

    everything = []
    for key, (title, cases) in sweeps.items():
        print("\n" + "="*78 + f"\nSWEEP {key}: {title}\n" + "="*78)
        results = [solve(c) for c in cases]
        for r in results:
            report(r)
        plot_sweep(results, f"dsi_v6_{key}.png", title)
        everything += results
    plot_summary(everything, "dsi_v6_summary.png")
