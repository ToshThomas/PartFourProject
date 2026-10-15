"""Whey-protein denaturation downstream of the injector (post-processor for dsi_2d_model.py).

The cross-sectional model gives the temperature of every layer along the pipe. This script
carries the whey protein through that field and asks whether the hot spots beside the
injector denature it.

Kinetics: the three-step model of Safavi Nic et al. (2020), fitted on a DSI pilot line
(report reference [14]):
    native <-> unfolded (k1, k-1),   unfolded + unfolded -> aggregated (k2, pseudo-second order)
    k = exp(ln k0 - Ea/RT), with k2 switching regime at 90 degC; refolding (k-1) applies up to 120 degC.
Unfolding and refolding are orders of magnitude faster than aggregation at the temperatures that
matter (k1, k-1 ~ 1e4 s^-1 at 72 degC), so native and unfolded protein are taken to be in local
equilibrium, unfolded fraction f_U = K/(1+K), K = k1/k-1 (f_U = 1 above 120 degC). Aggregation is
then  dc/dt = -k2 (f_U c)^2  for the native + unfolded concentration c, integrated exactly over each
axial step. Irreversibly aggregated protein is reported as "denatured".

Transport: native + unfolded and aggregated protein are carried as mass fractions in each layer,
diluted by the condensate the layer absorbs and exchanged between layers by the same implicit
lateral mixing as the heat (same conductances), so they follow exactly the paths of the energy.

The kinetics were fitted to a whey-enriched product, not milk, and under-predict published milk
data (printed below against Lee et al. 2017). Absolute denatured fractions are therefore not
reliable; the script reports each case RELATIVE TO THE SAME MILK HEATED UNIFORMLY (every layer at
the mixing-cup temperature). At low conversion that ratio depends on the activation energies and the
unfolding equilibrium, not on the pre-exponential factor or the protein concentration.

Writes results_denaturation.json and figs/report/fig_A5_6_denaturation.png next to this script.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R_GAS = 8.314
W_WHEY = 0.034*0.20      # whey protein mass fraction: 3.4 % protein, ~20 % of it whey (Munir et al. [17])
CASES = [
    ("top", "Top wall, 45° (base)", dict()),
    ("radial", "Top wall, 90°", dict(angle=90.0)),
    ("shallow", "Top wall, 20°", dict(angle=20.0)),
    ("centre", "Centre lance, co-axial", dict(position="centre", angle=0.0)),
    ("ring", "Ring, 4 nozzles, 45°", dict(position="ring", n_nozzles=4)),
]


def variant(**kw):
    """Load the model with module-level constants replaced (as in line_pressure_sweep.py)."""
    src = open(os.path.join(HERE, "dsi_2d_model.py")).read()
    for k, v in kw.items():
        src, n = re.subn(rf"^{k}\s*=.*$", f"{k} = {v!r}", src, count=1, flags=re.M)
        assert n == 1, k
    ns = {"__name__": "variant"}
    exec(compile(src, "variant", "exec"), ns)
    return ns


EQUILIBRIUM = True   # False: sensitivity case, all native protein treated as reactive (f_U = 1)


def kinetics(T):
    """Aggregation rate constant k2 (m3/kg.s) and unfolded fraction f_U, elementwise in T (degC)."""
    TK = np.asarray(T, dtype=float) + 273.15
    ln_k1 = 122.7 - 325.83e3/(R_GAS*TK)
    ln_km1 = 70.96 - 177.4e3/(R_GAS*TK)
    lnK = np.clip(ln_k1 - ln_km1, -700.0, 700.0)
    f_U = np.where(TK > 393.15, 1.0, 1.0/(1.0 + np.exp(-lnK)))
    k2 = np.where(TK < 363.15, np.exp(89.84 - 292.84e3/(R_GAS*TK)), np.exp(9.82 - 51.17e3/(R_GAS*TK)))
    if not EQUILIBRIUM:
        f_U = np.ones_like(f_U)
    return k2, f_U


def aggregate(c, T, dt):
    """Exact second-order step: native + unfolded concentration c (kg/m3) after time dt."""
    k2, f_U = kinetics(T)
    return c/(1.0 + k2*f_U**2*c*dt)


def thomas_cols(W, m, kf):
    """thomas_diffuse from the model, for several columns at once (W is ny x k)."""
    n = len(m)
    a = np.zeros(n); b = m.copy(); c = np.zeros(n)
    b[:-1] += kf; b[1:] += kf
    a[1:] = -kf; c[:-1] = -kf
    d = m[:, None]*W
    cp_ = np.zeros(n); dp = np.zeros_like(W)
    cp_[0] = c[0]/b[0]; dp[0] = d[0]/b[0]
    for j in range(1, n):
        den = b[j] - a[j]*cp_[j-1]
        cp_[j] = c[j]/den
        dp[j] = (d[j] - a[j]*dp[j-1])/den
    x = np.zeros_like(W); x[-1] = dp[-1]
    for j in range(n-2, -1, -1):
        x[j] = dp[j] - cp_[j]*x[j+1]
    return x


def march(V, T_field, m_field, D_x):
    """Carry native+unfolded (column 0) and aggregated (column 1) whey mass fractions along the pipe."""
    rho, A_cell, dx, dy, L_iface = V["rho"], V["A_cell"], V["dx"], V["dy"], V["L_iface"]
    nx, ny = T_field.shape
    W = np.zeros((ny, 2)); W[:, 0] = W_WHEY
    agg_frac = np.zeros((nx, ny))
    for i in range(1, nx):
        W *= (m_field[i-1]/m_field[i])[:, None]                 # dilution by absorbed condensate
        dt = dx*rho*A_cell/m_field[i]                           # residence time of each layer in the step
        c_old = W[:, 0]*rho
        c_new = aggregate(c_old, T_field[i], dt)
        W[:, 0] = c_new/rho
        W[:, 1] += (c_old - c_new)/rho
        kf = rho*D_x[i]*L_iface*dx/dy
        W = thomas_cols(W, m_field[i], kf)
        agg_frac[i] = W[:, 1]/W.sum(axis=1)
    return agg_frac


def cup(field, m_field):
    return (field*m_field).sum(axis=1)/m_field.sum(axis=1)


def benchmark():
    """Uniform-temperature predictions for the conditions measured by Lee et al. (2017), report ref [21]."""
    c0 = W_WHEY*1000.0
    out = {}
    for label, T, t, measured in [("HTST, 78 degC for 15 s", 78.0, 15.0, 20.0),
                                  ("DSI-UP, 140 degC for 2.3 s", 140.0, 2.3, 70.9)]:
        out[label] = dict(predicted_pct=100*(1 - aggregate(c0, T, t)/c0), measured_SPDN_pct=measured)
    return out


if __name__ == "__main__":
    V = variant()
    results = {"benchmark_against_Lee_2017": benchmark(), "whey_mass_fraction_in": W_WHEY,
               "note": ("injector excess = mixing-cup aggregation at the 41 m outlet minus that of the same milk "
                        "heated uniformly at the mixing-cup temperature, as % of the uniform value and as the "
                        "equivalent extra time at 72 degC (scaled by the mean hold residence time). "
                        "'equilibrium' = native/unfolded equilibrium of the fitted kinetics (base); "
                        "'all_reactive' = sensitivity case with every native molecule reactive (f_U = 1).")}
    curves = {}
    for mode in ("equilibrium", "all_reactive"):
        EQUILIBRIUM = (mode == "equilibrium")
        results[mode] = {}
        for key, label, kw in CASES:
            r = V["solve"](V["Case"](label, **kw))
            T, m = r["T"], r["m"]
            cup_case = cup(march(V, T, m, r["D_x"]), m)
            T_uni = np.repeat(r["Tb"][:, None], T.shape[1], axis=1)
            cup_ref = cup(march(V, T_uni, m, r["D_x"]), m)
            hold_total = cup_ref[-1]
            excess = (cup_case[-1] - hold_total)/hold_total
            results[mode][key] = dict(
                label=label,
                injector_excess_pct_of_hold=float(100*excess),
                injector_excess_equiv_s_at_72C=float(excess*r["t_mean"]),
                outlet_aggregated_pct=float(100*cup_case[-1]),
                outlet_aggregated_pct_uniform=float(100*hold_total),
                hold_mean_residence_s=float(r["t_mean"]),
                peak_wall_layer_T=float(max(r["top"].max(), r["bot"].max())),
                mixing_length_mm=float(1000*r["L_mix"]),
            )
            if mode == "equilibrium":
                curves[key] = (label, 100*(cup_case - cup_ref)/hold_total)
            print(f"[{mode:12s}] {label:24s} injector adds {100*excess:5.1f} % of the hold's aggregation "
                  f"(= {excess*r['t_mean']:.2f} s at 72 degC)")

    print("Benchmark against Lee et al. (2017), uniform temperature, base kinetics:")
    EQUILIBRIUM = True
    for k, v in results["benchmark_against_Lee_2017"].items():
        print(f"  {k}: predicted {v['predicted_pct']:.2f} %, measured SPDN {v['measured_SPDN_pct']:.1f} %")
    with open(os.path.join(HERE, "results_denaturation.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # Figure: (a) aggregation added by the injector along the first 0.15 m (base kinetics);
    #         (b) total added at the outlet for both kinetic treatments.
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.2, 1]))
    x = V["x_axis"] - V["inj_position"]
    sel = (x >= -0.01) & (x <= 0.15)
    for key, (label, excess_pct) in curves.items():
        ax1.plot(1000*x[sel], excess_pct[sel], label=label)
    ax1.set_xlabel("Distance from injector (mm)")
    ax1.set_ylabel("Aggregation added by injector\n(% of whole-hold value)")
    ax1.legend(fontsize=7, frameon=False)
    keys = [k for k, _, _ in CASES]
    idx = np.arange(len(keys))
    ax2.barh(idx - 0.2, [results["equilibrium"][k]["injector_excess_pct_of_hold"] for k in keys], 0.4,
             label="Fitted kinetics (base)")
    ax2.barh(idx + 0.2, [results["all_reactive"][k]["injector_excess_pct_of_hold"] for k in keys], 0.4,
             label="All protein reactive")
    ax2.set_yticks(idx); ax2.set_yticklabels([lab for _, lab, _ in CASES], fontsize=7); ax2.invert_yaxis()
    ax2.set_xlabel("Added at outlet (% of whole-hold value)")
    ax2.legend(fontsize=7, frameon=False, loc="lower right")
    ax1.set_title("(a)", loc="left"); ax2.set_title("(b)", loc="left")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figs", "report"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figs", "report", "fig_A5_6_denaturation.png"), dpi=200)
    print("saved figs/report/fig_A5_6_denaturation.png")
