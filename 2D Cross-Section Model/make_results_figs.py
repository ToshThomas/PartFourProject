"""Results figures for the report (Figures 4.1-4.7 and A5.1-A5.4), in one consistent style.

Also writes results.json (case summaries, grid study for Table A4.2, sensitivity for Table A4.3);
the poster script reads its "sens" block.

Re-runs the cross-sectional model (same inputs as the report) and reads the 1D node
table from the workbook. Writes to figs/report/, one file per report figure.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

import dsi_2d_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs", "report")
WORKBOOK = os.path.join(HERE, "..", "1D Excel Model", "DSI 1D Model_updated.xlsx")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.linewidth": 0.6, "axes.labelsize": 9, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "legend.frameon": False, "legend.fontsize": 8,
    "legend.handlelength": 2.6, "axes.grid": False, "savefig.dpi": 300,
})
# coloured series for the graphs, so each case is easy to follow
SERIES = [dict(color="tab:blue", lw=1.4), dict(color="tab:orange", lw=1.4), dict(color="tab:green", lw=1.4),
          dict(color="tab:red", lw=1.4), dict(color="tab:purple", lw=1.4)]
REF = dict(color="k", lw=0.7, ls=":")            # 72 °C / 1 °C reference lines
CAP = dict(color="0.45", lw=0.8, ls="--")        # saturation cap
CMAP = "inferno"
X0, X1 = M.inj_position, M.inj_position + 0.35    # injector-site window for fields
XL = M.inj_position + 0.8                         # window for line plots


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def tag(ax, letter, x=0.02, y=0.96, color="k"):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, va="top", ha="left", fontsize=9, color=color)


def tag_above(ax, letter):
    """Panel letter above the top-left corner of a line plot, clear of the data."""
    ax.set_title(f"({letter})", loc="left", fontsize=9, pad=3)


def log_spread(ax):
    ax.set_yscale("log"); ax.set_ylim(0.3, 200)
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100])); ax.set_yticklabels(["1", "10", "100"])
    ax.yaxis.set_minor_formatter(NullFormatter())


def field(ax, res, vmax=None):
    vmax = vmax or M.T_sat_line
    i0, i1 = int((X0 - 0.05)/M.dx), int(X1/M.dx)
    pcm = ax.pcolormesh(M.x_axis[i0:i1], M.y*1000, res['T'][i0:i1].T, shading="auto", cmap=CMAP,
                        vmin=M.milk_T_in, vmax=vmax, rasterized=True)
    for t in res['trajs']:
        ax.plot(t['x'], t['y']*1000, color="white", lw=0.8, ls=(0, (4, 2)))
    ax.set_xlim(X0 - 0.05, X1); ax.set_ylim(0, 50)
    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("Height (mm)")
    ax.tick_params(colors="k", which="both"); ax.tick_params(axis="both", which="both", color="w")
    return pcm


def cbar(fig, mappable, ax, **kw):
    cb = fig.colorbar(mappable, ax=ax, extend="max", pad=0.02, **kw)
    cb.set_label("$T$ (°C)"); cb.outline.set_linewidth(0.6); cb.ax.tick_params(direction="in", width=0.6)
    return cb


def window(ax):
    ax.set_xlim(X0 - 0.05, XL); ax.set_xlabel("$x$ (m)")


# ------------------------------------------------------------------ model runs
base = M.solve(M.Case("base"))
basis = M.solve(M.Case("basis", nozzle="basis"))
pos = [("Top wall (base)", M.Case("top", position="top")), ("Bottom wall", M.Case("bottom", position="bottom")),
       ("Centre lance", M.Case("centre", position="centre", angle=0.0)), ("Ring, 4 nozzles", M.Case("ring", position="ring", n_nozzles=4))]
rp = {lab: M.solve(c) for lab, c in pos}
angles = [90, 45, 20]
ra = {a: M.solve(M.Case(f"{a}", angle=a)) for a in angles}
walls = [("Adiabatic", "adiabatic"), ("Bare stainless", "bare"), ("50 mm insulation", "insulated")]
rw = {lab: M.solve(M.Case(lab, wall=w)) for lab, w in walls}
i1 = int(XL/M.dx); x = M.x_axis[:i1]

# ------------------------------------------------------------------ Figure 4.1 nozzle capacity
d = np.linspace(2, 35, 200)/1000
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(d*1000, M.G_choked*np.pi*(d/2)**2*3600, color="steelblue", lw=1.8, label="Choked capacity, 6 bar(a), $C_d$ = 0.85")
ax.axhline(M.mdot_req*3600, color="crimson", ls="--", lw=1.2, label=f"Required for 72 °C ({M.mdot_req*3600:,.0f} kg/h)")
for dv in (5, M.d_required_single*1000):
    ax.axvline(dv, **REF)
ax.annotate(f"5 mm: {M.mdot_basis_single*3600:.0f} kg/h", (5, M.mdot_basis_single*3600), xytext=(6.5, 650),
            arrowprops=dict(arrowstyle="-", lw=0.5))
ax.annotate(f"{M.d_required_single*1000:.1f} mm", (M.d_required_single*1000, M.mdot_req*3600), xytext=(29.5, 850),
            arrowprops=dict(arrowstyle="-", lw=0.5))
ax.set_xlim(0, 36); ax.set_ylim(0, 2700)
ax.set_xlabel("Nozzle bore (mm)"); ax.set_ylabel("Steam mass flow (kg/h)"); ax.legend(loc="upper left")
save(fig, "fig_4_1_nozzle_capacity.png")

# ------------------------------------------------------------------ Figure 4.2 base case
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
p = field(axes[0], base); cbar(fig, p, axes[0]); tag(axes[0], "a", color="w")
ax = axes[1]
for (lab, y), st in zip([("Mixing-cup", base['Tb']), ("Top wall layer", base['top']), ("Bottom wall layer", base['bot']),
                          ("Centre layer", base['T'][:, M.ny//2])],
                         [dict(color="crimson", lw=1.8), dict(color="darkorange", lw=1.2),
                          dict(color="steelblue", lw=1.2), dict(color="grey", lw=1.0)]):
    ax.plot(x, y[:i1], label=lab, **st)
ax.axhline(M.T_target, **REF); ax.axhline(M.T_sat_line, **CAP)
window(ax); ax.set_ylabel("$T$ (°C)"); ax.set_ylim(0, 140); ax.legend(loc="center right"); tag_above(ax, "b")
plt.tight_layout(); save(fig, "fig_4_2_base_case.png")

# ------------------------------------------------------------------ Figure 4.3 position fields
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
for ax, (lab, _), k in zip(axes.flat, pos, "abcd"):
    p = field(ax, rp[lab]); tag(ax, k, color="w")
cbar(fig, p, axes, shrink=0.8)
save(fig, "fig_4_3_position_fields.png")

# ------------------------------------------------------------------ Figure 4.4 position lines
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for (lab, _), st in zip(pos, SERIES):
    r = rp[lab]
    st = dict(st, ls="--") if lab == "Bottom wall" else st     # identical to the top wall (no buoyancy); dashed so both show
    axes[0].plot(x, r['spread'][:i1], label=lab, **st)
    axes[1].plot(x, np.maximum(r['top'], r['bot'])[:i1], label=lab, **st)
log_spread(axes[0]); axes[0].axhline(1, **REF)
axes[0].set_ylabel("Cross-section spread (°C)"); axes[0].legend(loc="upper right")
axes[1].axhline(M.T_target, **REF); axes[1].set_ylabel("Hotter wall layer, $T$ (°C)")
for ax, k in zip(axes, "ab"):
    window(ax); tag_above(ax, k)
plt.tight_layout(); save(fig, "fig_4_4_position_lines.png")

# ------------------------------------------------------------------ Figure 4.5 angle fields
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)
for ax, a, k in zip(axes, angles, "abc"):
    p = field(ax, ra[a]); tag(ax, k, color="w")
    if k != "a":
        ax.set_ylabel(""); ax.set_yticklabels([])
cbar(fig, p, axes, shrink=0.9)
save(fig, "fig_4_5_angle_fields.png")

# ------------------------------------------------------------------ Figure 4.6 heat loss
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for (lab, _), st in zip(walls, [dict(color="k", lw=1.4), dict(color="crimson", lw=1.4), dict(color="steelblue", lw=1.4)]):
    r = rw[lab]
    axes[0].plot(M.x_axis, r['Tb'], label=lab, **st)
    axes[1].plot(M.x_axis, r['T'][:, 0], label=lab, **st)
axes[0].set_ylabel("Mixing-cup $T$ (°C)"); axes[1].set_ylabel("Bottom wall layer, $T$ (°C)")
axes[0].legend(loc="lower left")
for ax, k in zip(axes, "ab"):
    ax.axhline(M.T_target, **REF); ax.set_xlim(0, M.pipe_length); ax.set_ylim(71.6, 72.1)
    ax.set_xlabel("$x$ (m)"); tag_above(ax, k)
plt.tight_layout(); save(fig, "fig_4_6_heat_loss.png")

# ------------------------------------------------------------------ Figure 4.7 summary
allres = [("5 mm, top 45°", basis), ("Top 45° (base)", base), ("Top 90°", ra[90]), ("Top 20°", ra[20]),
          ("Bottom 45°", rp["Bottom wall"]), ("Centre lance 0°", rp["Centre lance"]), ("Ring ×4, 45°", rp["Ring, 4 nozzles"])]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6)); idx = np.arange(len(allres))
bar = dict(height=0.65)
axes[0].barh(idx, [r['L_mix']*1000 for _, r in allres], color="steelblue", **bar); axes[0].set_xlabel("Mixing length to < 1 °C (mm)")
axes[1].barh(idx, [max(r['top'].max(), r['bot'].max()) for _, r in allres], color="darkorange", **bar); axes[1].axvline(M.T_target, **REF)
axes[1].set_xlabel("Peak wall-layer $T$ (°C)")
axes[2].barh(idx, [r['n_sat'] for _, r in allres], color="crimson", **bar); axes[2].set_xlabel("Layers at $T_{sat}$ (of 40)")
for ax in axes:
    ax.set_yticks(idx); ax.set_yticklabels([l for l, _ in allres]); ax.invert_yaxis()
    ax.tick_params(axis="y", which="both", left=False, right=False)
for ax in axes[1:]:
    ax.set_yticklabels([])
plt.tight_layout(); save(fig, "fig_4_7_summary.png")

# ------------------------------------------------------------------ Figure A5.1 1D node table
ws = openpyxl.load_workbook(WORKBOOK, data_only=True)["3 Node Calculations"]
rows = [r for r in ws.iter_rows(min_row=19, values_only=True) if isinstance(r[0], (int, float))]
xn = np.array([r[1] for r in rows]); Tn = np.array([r[11] for r in rows]); wn = np.array([r[8] for r in rows])
fig, ax = plt.subplots(figsize=(5.2, 2.8))
l1 = ax.step(xn, Tn, where="post", color="crimson", lw=1.6, label="Temperature")
ax.set_xlabel("Distance from inlet (m)"); ax.set_ylabel("$T$ (°C)"); ax.set_ylim(0, 90); ax.set_xlim(0, xn.max())
ax.tick_params(right=False)
ax2 = ax.twinx(); ax2.tick_params(direction="in", width=0.6)
l2 = ax2.step(xn, wn, where="post", color="steelblue", lw=1.4, ls="--", label="Solids mass fraction")
ax2.set_ylabel("Solids mass fraction (kg/kg)"); ax2.set_ylim(0.100, 0.140)
ax.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="center right")
plt.tight_layout(); save(fig, "fig_A5_1_1d_profile.png")

# ------------------------------------------------------------------ Figure A5.2 nozzle fields
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
p1 = field(axes[0], basis, vmax=40); cbar(fig, p1, axes[0]); tag(axes[0], "a", color="w")
p2 = field(axes[1], base); cbar(fig, p2, axes[1]); tag(axes[1], "b", color="w")
plt.tight_layout(); save(fig, "fig_A5_2_nozzle_fields.png")

# ------------------------------------------------------------------ Figure A5.3 angle lines
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for a, st in zip(angles, SERIES):
    r = ra[a]
    axes[0].plot(x, r['spread'][:i1], label=f"{a}°", **st)
    axes[1].plot(x, r['bot'][:i1], label=f"{a}°", **st)
log_spread(axes[0]); axes[0].axhline(1, **REF)
axes[0].set_ylabel("Cross-section spread (°C)"); axes[0].legend(loc="upper right")
axes[1].axhline(M.T_target, **REF); axes[1].set_ylabel("Far-wall (bottom) layer, $T$ (°C)")
for ax, k in zip(axes, "ab"):
    window(ax); tag_above(ax, k)
plt.tight_layout(); save(fig, "fig_A5_3_angle_lines.png")

# ------------------------------------------------------------------ Figure A5.4 sensitivity to the mixing constants
import re


def variant(**kw):
    src = open(os.path.join(HERE, "dsi_2d_model.py")).read()
    for k, v in kw.items():
        src = re.sub(rf"^{k}\s*=.*$", f"{k} = {v!r}", src, count=1, flags=re.M)
    ns = {"__name__": "variant"}; exec(compile(src, "variant", "exec"), ns); return ns


sens = []
for cj in [0.0125, 0.025, 0.05, 0.1, 0.2]:
    for lm in [2.5, 5.0, 10.0]:
        V = variant(C_jet=cj, L_mix_D=lm, pipe_length=2.0)
        rt = V["solve"](V["Case"]("top")); rc = V["solve"](V["Case"]("c", position="centre", angle=0.0))
        sens.append(dict(C_jet=cj, L_mix_D=lm, spread_top=rt['spread'][-1], Lmix_top=rt['L_mix'],
                         spread_c=rc['spread'][-1], Lmix_c=rc['L_mix'], over70_top=rt['over70_bot']))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for lm, st, mk in zip([2.5, 5.0, 10.0], SERIES, ["o", "s", "^"]):
    rows_ = [r for r in sens if r['L_mix_D'] == lm]
    cj = [r['C_jet'] for r in rows_]
    axes[0].plot(cj, [r['spread_top'] for r in rows_], marker=mk, ms=4, label=f"$L_{{mix}}$ = {lm:g} $D$", **st)
    axes[1].plot(cj, [r['Lmix_c']*1000 for r in rows_], marker=mk, ms=4, **st)
for ax, k in zip(axes, "ab"):
    ax.set_xscale("log"); ax.set_xlabel("$C_{jet}$"); tag_above(ax, k)
    ax.xaxis.set_major_locator(FixedLocator([0.0125, 0.025, 0.05, 0.1, 0.2]))
    ax.set_xticklabels(["0.0125", "0.025", "0.05", "0.1", "0.2"]); ax.xaxis.set_minor_locator(NullLocator())
axes[0].set_ylabel("Outlet spread, top-wall injector (°C)"); axes[0].legend(loc="center right")
axes[1].set_ylabel("Mixing length, centre lance (mm)")
plt.tight_layout(); save(fig, "fig_A5_4_sensitivity_constants.png")

# ------------------------------------------------------------------ results.json
# Grid study (Table A4.2): first 2 m of pipe at (dx in mm, layers).
grid = []
for dx_mm, n_y in [(2, 20), (1, 40), (0.5, 80), (1, 20), (1, 80), (2, 40), (0.5, 40)]:
    V = variant(pipe_length=2.0, dx_target=dx_mm*1e-3, ny=n_y)
    r = V["solve"](V["Case"]("g"))
    grid.append(dict(dx_mm=dx_mm, ny=n_y, spread=r['spread'][-1], Lmix=r['L_mix'], peak=r['top'].max(),
                     over70=r['over70_bot'], spread_05=r['spread'][int(0.5/V['dx'])]))


def jet_summary(r):
    t = r['trajs'][0]
    return dict(pen=t['pen']/M.pipe_diameter, end=t['end'], x_end=t['x'][-1] - M.inj_position)


results = dict(
    basis=dict(T_out=basis['T_out'], spread=basis['spread'][-1], Lmix=basis['L_mix'], top=basis['top'].max(), **jet_summary(basis)),
    required=dict(T_out=base['T_out'], spread=base['spread'][-1], Lmix=base['L_mix'], top=base['top'].max(),
                  nsat=base['n_sat'], over70=base['over70_bot'], **jet_summary(base)),
    angle={str(a): dict(spread=r['spread'][-1], Lmix=r['L_mix'], nsat=r['n_sat'], pen=jet_summary(r)['pen'],
                        x_end=jet_summary(r)['x_end']) for a, r in ra.items()},
    position={key: dict(spread=r['spread'][-1], Lmix=r['L_mix'], nsat=r['n_sat'], peak=max(r['top'].max(), r['bot'].max()),
                        over70=max(r['over70_top'], r['over70_bot']))
              for key, (lab, _) in zip(["top", "bottom", "centre", "ring"], pos) for r in [rp[lab]]},
    wall={w: dict(T_out=r['T_out'], Q=r['Q_loss']*1000, U=r['U_w'], spread=r['spread'][-1])
          for (lab, w) in walls for r in [rw[lab]]},
    grid=grid,
    sens=sens,
)
import json
json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
print("wrote results.json")
