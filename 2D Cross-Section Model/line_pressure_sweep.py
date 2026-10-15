"""Sensitivity of the cross-sectional model to the milk-line pressure.

The line pressure enters the model only through the saturation cap
(T_sat at the line pressure). Each case is re-run over the first 2 m of
pipe (all jet physics, adiabatic wall) at line pressures up to the choking
limit of the 6 bar(a) supply (about 0.58 x 6 = 3.5 bar(a)), so the injector
stays choked in every case and only the cap changes.

Writes results_line_pressure.json next to this script.
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LINE_PRESSURES = [1.5, 2.0, 2.5, 3.0, 3.4]          # bar(a); 3.0 is the base value
CASES = [
    ("Top wall, 45° (base)", dict()),
    ("Top wall, 90° (radial)", dict(angle=90.0)),
    ("Centre lance, co-axial", dict(position="centre", angle=0.0)),
    ("Ring, 4 nozzles, 45°", dict(position="ring", n_nozzles=4)),
]


def variant(**kw):
    """Load the model with module-level constants replaced (same approach as make_results_figs.py)."""
    src = open(os.path.join(HERE, "dsi_2d_model.py")).read()
    for k, v in kw.items():
        src, n = re.subn(rf"^{k}\s*=.*$", f"{k} = {v!r}", src, count=1, flags=re.M)
        assert n == 1, k
    ns = {"__name__": "variant"}
    exec(compile(src, "variant", "exec"), ns)
    return ns


rows = []
for p in LINE_PRESSURES:
    V = variant(line_pressure=p * 1e5, pipe_length=2.0)
    for label, kw in CASES:
        r = V["solve"](V["Case"](label, **kw))
        rows.append(dict(
            line_bar=p, T_sat_line=float(V["T_sat_line"]), case=label,
            n_sat=int(r["n_sat"]),
            peak_wall=float(max(r["top"].max(), r["bot"].max())),
            hot_len_mm=float(max(r["hot_top"], r["hot_bot"]) * 1000),
            L_mix_mm=float(r["L_mix"] * 1000) if np.isfinite(r["L_mix"]) else None,
            T_out=float(r["T_out"]),
            uncondensed=float(r["uncondensed"]),
        ))
        print(f"{p:4.1f} bar  T_sat {rows[-1]['T_sat_line']:6.1f}  {label:24s}  n_sat {rows[-1]['n_sat']:2d}  "
              f"peak wall {rows[-1]['peak_wall']:6.1f}  wall>product {rows[-1]['hot_len_mm']:5.0f} mm  "
              f"L_mix {rows[-1]['L_mix_mm']}  T_out {rows[-1]['T_out']:.2f}")

json.dump(rows, open(os.path.join(HERE, "results_line_pressure.json"), "w"), indent=1)
