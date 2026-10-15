"""Propagated uncertainty of the cross-sectional model (Monte Carlo over the uncertain inputs).

The three inputs that were not measured are sampled together by Latin hypercube:
  C_jet          log-uniform over 0.0125-0.2 (the sixteen-fold range of the sensitivity study)
  L_mix_D        uniform over 2.5-10 pipe diameters
  line_pressure  uniform over 1.5-3.4 bar(a) (up to the choking limit of the 6 bar(a) supply)
Each sample runs four injector configurations over the first 2 m of pipe (adiabatic wall).
The ranges are assumed bounds, not measured distributions, so the percentiles describe how far
the outputs move within those bounds; they are not statistical confidence intervals.

Writes results_uncertainty.json and figs/report/fig_A5_5_uncertainty.png next to this script.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_SAMPLES = 300
SEED = 2026
CASES = [
    ("top", "Top wall, 45° (base)", dict()),
    ("radial", "Top wall, 90°", dict(angle=90.0)),
    ("centre", "Centre lance, co-axial", dict(position="centre", angle=0.0)),
    ("ring", "Ring, 4 nozzles, 45°", dict(position="ring", n_nozzles=4)),
]
SRC = open(os.path.join(HERE, "dsi_2d_model_v6.py")).read()


def variant(**kw):
    """Load the model with module-level constants replaced (as in line_pressure_sweep.py)."""
    src = SRC
    for k, v in kw.items():
        src, n = re.subn(rf"^{k}\s*=.*$", f"{k} = {v!r}", src, count=1, flags=re.M)
        assert n == 1, k
    ns = {"__name__": "variant"}
    exec(compile(src, "variant", "exec"), ns)
    return ns


def latin_hypercube(n, d, rng):
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return u


rng = np.random.default_rng(SEED)
u = latin_hypercube(N_SAMPLES, 3, rng)
samples = np.column_stack([
    0.0125 * 16 ** u[:, 0],          # C_jet, log-uniform 0.0125-0.2
    2.5 + 7.5 * u[:, 1],             # L_mix_D, uniform 2.5-10
    1.5 + 1.9 * u[:, 2],             # line pressure, bar(a), uniform 1.5-3.4
])

rows = []
for i, (cj, lm, pl) in enumerate(samples):
    V = variant(C_jet=float(cj), L_mix_D=float(lm), line_pressure=float(pl) * 1e5, pipe_length=2.0)
    row = dict(C_jet=float(cj), L_mix_D=float(lm), line_bar=float(pl))
    for key, label, kw in CASES:
        r = V["solve"](V["Case"](label, **kw))
        row[key] = dict(
            L_mix_mm=float(r["L_mix"] * 1000) if np.isfinite(r["L_mix"]) else None,
            spread_2m=float(r["spread"][-1]),
            peak_wall=float(max(r["top"].max(), r["bot"].max())),
            n_sat=int(r["n_sat"]),
            hot_len_mm=float(max(r["hot_top"], r["hot_bot"]) * 1000),
            T_out=float(r["T_out"]),
        )
    rows.append(row)
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{N_SAMPLES} samples")


def pct(vals, q):
    return float(np.percentile(vals, q))


summary = {}
for key, label, _ in CASES:
    lm = np.array([r[key]["L_mix_mm"] if r[key]["L_mix_mm"] is not None else np.inf for r in rows])
    mixed = np.isfinite(lm)
    pw = np.array([r[key]["peak_wall"] for r in rows])
    ns = np.array([r[key]["n_sat"] for r in rows])
    hl = np.array([r[key]["hot_len_mm"] for r in rows])
    summary[key] = dict(
        label=label,
        frac_mixed_2m=float(mixed.mean()),
        L_mix_mm_p5_p50_p95=[pct(lm[mixed], q) for q in (5, 50, 95)] if mixed.any() else None,
        peak_wall_p5_p50_p95=[pct(pw, q) for q in (5, 50, 95)],
        n_sat_min_median_max=[int(ns.min()), float(np.median(ns)), int(ns.max())],
        hot_len_mm_p5_p50_p95=[pct(hl, q) for q in (5, 50, 95)],
        wall_above_product_frac=float((hl > 0).mean()),
    )

# ranking checks, sample by sample
lance_faster = [ (r["centre"]["L_mix_mm"] or np.inf) < (r["top"]["L_mix_mm"] or np.inf) for r in rows]
lance_cool = [r["centre"]["hot_len_mm"] == 0.0 for r in rows]
walls_hot = [r["top"]["hot_len_mm"] > 0 and r["ring"]["hot_len_mm"] > 0 and r["radial"]["hot_len_mm"] > 0 for r in rows]
summary["ranking"] = dict(
    lance_mixes_faster_than_top_frac=float(np.mean(lance_faster)),
    lance_wall_never_above_product_frac=float(np.mean(lance_cool)),
    wall_and_ring_injectors_heat_wall_frac=float(np.mean(walls_hot)),
)
summary["design"] = dict(n_samples=N_SAMPLES, seed=SEED, C_jet="log-uniform 0.0125-0.2",
                         L_mix_D="uniform 2.5-10", line_pressure_bar="uniform 1.5-3.4",
                         domain="first 2 m, adiabatic wall, 27.5 mm nozzle")

json.dump(dict(summary=summary, samples=rows), open(os.path.join(HERE, "results_uncertainty.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

# ---------------------------------------------------------------- figure (report style)
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.linewidth": 0.6, "axes.labelsize": 9, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 300,
})
colors = {"top": "tab:blue", "radial": "tab:orange", "centre": "tab:green", "ring": "tab:red"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for key, label, _ in CASES:
    lm = np.sort([r[key]["L_mix_mm"] for r in rows if r[key]["L_mix_mm"] is not None])
    axes[0].step(lm, np.arange(1, len(lm) + 1) / N_SAMPLES, where="post", color=colors[key], lw=1.4, label=label)
    pw = np.sort([r[key]["peak_wall"] for r in rows])
    axes[1].step(pw, np.arange(1, len(pw) + 1) / N_SAMPLES, where="post", color=colors[key], lw=1.4, label=label)
axes[0].set_xscale("log")
axes[0].set_xlabel("Mixing length (mm)")
axes[0].set_ylabel("Fraction of samples")
axes[1].axvline(72.0, color="k", lw=0.7, ls=":")
axes[1].set_xlabel("Peak wall-layer temperature (°C)")
axes[1].set_ylabel("Fraction of samples")
for ax, letter in zip(axes, "ab"):
    ax.set_ylim(0, 1.02)
    ax.set_title(f"({letter})", loc="left", fontsize=9, pad=3)
axes[0].legend(loc="lower right")
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figs", "report"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "report", "fig_A5_5_uncertainty.png"), bbox_inches="tight")
print("wrote figs/report/fig_A5_5_uncertainty.png")
