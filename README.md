# Modelling of Direct Steam Injection for Milk Heating: model compendium

CHEMMAT 751 Final Year Research Project, University of Auckland, 2026
Tosh Thomas. Supervisors: Isaac Severinsen and Brent Young.

This compendium contains the two models described in the final report and the scripts that produce every number, table and figure in it. Each result in the report can be regenerated from the files here.

## Contents

```
DSI Model Compendium/
├── README.md                      this file
├── requirements.txt               Python packages needed
├── 1D Excel Model/
│   └── DSI 1D Model_updated.xlsx  one-dimensional mass, solids and energy balance
└── 2D Cross-Section Model/
    ├── dsi_2d_model.py            cross-sectional (layered) model, all case sweeps
    ├── run_log_final.txt          output of dsi_2d_model.py used in the report
    ├── make_results_figs.py       report Figures 4.1–4.4 and A4.1–A4.7; writes results.json
    ├── make_fig_schematics.py     report Figures 1.1 and 3.1 (schematics)
    ├── line_pressure_sweep.py     line-pressure sensitivity; writes results_line_pressure.json
    ├── uncertainty_mc.py          Monte Carlo propagated uncertainty; writes results_uncertainty.json and Figure A4.8
    ├── denaturation.py            whey-protein denaturation post-processor; writes results_denaturation.json and Figure 4.5
    ├── results.json               case summaries, grid study and mixing-constant sensitivity
    ├── results_line_pressure.json line-pressure sensitivity results
    ├── results_uncertainty.json   Monte Carlo samples and summary (300 samples, seed 2026)
    ├── results_denaturation.json  denaturation added by each injector, two kinetic treatments
    ├── run_log_denaturation.txt   output of denaturation.py used in the report
    ├── dsi_2d_*.png               quick-look plots written by dsi_2d_model.py (not in the report)
    └── figs/
        ├── fig_dsi_schematic.png, fig_model_structure.png
        └── report/                the report's results figures (file names follow an earlier numbering; see the table below)
```

## The two models

**1D Excel model.** A node-by-node steady-state balance along the pipe (100 nodes at 0.02 m). It sizes the steam demand for the 72 °C target, tracks the dilution of the milk solids and checks the choked capacity of the nozzle. Every input is on sheet 1 and is referenced by formula elsewhere. Sheet 6 documents every equation and assumption.

| Sheet | Contents |
|---|---|
| 1 Inputs | All design inputs (report Table A6.1) |
| 2 Steam Properties | IAPWS-IF97 saturated steam table and interpolation (report Table A1.1) |
| 3 Node Calculations | The node march, with an energy-closure column that must read zero (report Table A2.1) |
| 4 Results | Outlet conditions, evaporation duty, flow regime and nozzle checks (report Table 4.1) |
| 5 Steam Sizing | Steam demand for a target temperature (report Eq. 3.4) |
| 6 Documentation | Equations, assumptions and extension routes |

**2D cross-sectional model.** A Python script that divides the pipe into 40 horizontal layers and marches them along a 41 m pipe in 1 mm steps. It combines a choked-nozzle limit, a condensation length, an integral jet model, jet-enhanced turbulent mixing, a saturation cap at the line pressure and wall heat loss. All inputs sit in one block at the head of the script. Its equations, assumptions and sources are documented in the final report (Section 3.4 and Appendix V).

## Reproducing the results

Requires Python 3.10 or later. The results were produced with Python 3.14.6, NumPy 2.5.4, Matplotlib 3.11.2 and openpyxl 3.1.5. The workbook was built in Microsoft Excel 16.92 for Mac and contains formulas only, no macros.

```
pip install -r requirements.txt
cd "2D Cross-Section Model"
python dsi_2d_model.py > run_log_final.txt   # all case sweeps, about 45 s
python make_results_figs.py                         # report figures and results.json, about 30 s
python line_pressure_sweep.py                       # line-pressure sensitivity, about 3 s
python uncertainty_mc.py                            # propagated uncertainty, about 2 min
python denaturation.py > run_log_denaturation.txt   # whey-protein denaturation, about 3 min
python make_fig_schematics.py                       # Figures 1.1 and 3.1
```

`make_results_figs.py` reads the 1D workbook for Figure A4.1, so keep the two model folders side by side. Re-running the scripts reproduces `run_log_final.txt`, `results.json`, `results_line_pressure.json`, `results_uncertainty.json` and `results_denaturation.json` exactly (the Monte Carlo uses a fixed seed). This was checked on 14 October 2026.

## Where each report result comes from

| Report item | Source |
|---|---|
| Table A6.1 (design inputs) | Workbook sheet 1; input block of `dsi_2d_model.py` |
| Table 4.1 (1D baseline) | Workbook sheets 4 and 5 |
| Table A3.5 (all injector configurations) | `run_log_final.txt` (sweeps 1–3); `results.json` |
| Table A3.6 (wall heat loss) | `run_log_final.txt` (sweep 4); `results.json` (`wall`) |
| Table A1.1 (steam table) | Workbook sheet 2 |
| Table A2.1 (node table extract) | Workbook sheet 3 |
| Table A3.1 (verification) | Workbook energy-closure column; `run_log_final.txt` |
| Table A3.2 (grid study) | `results.json` (`grid`) |
| Table A3.3 (mixing-constant sensitivity) | `results.json` (`sens`) |
| Table A3.4 (line-pressure sensitivity) | `results_line_pressure.json` |
| Table A3.7 and Figure A4.8 (propagated uncertainty) | `uncertainty_mc.py`; `results_uncertainty.json` |
| Figures 1.1 and 3.1 | `make_fig_schematics.py` |
| Figures 4.1–4.4 and A4.1–A4.7 | `make_results_figs.py` |
| Figure A4.8 | `uncertainty_mc.py` |
| Sections 3.4.7, 4.8 and 5.4 (whey-protein denaturation) and Figure 4.5 | `denaturation.py`; `results_denaturation.json` |

## Changing the inputs

To run a new case in the 1D model, change the values on sheet 1. The rest of the workbook updates by formula. In the 2D model, edit the input block at the head of `dsi_2d_model.py`, or add a `Case(...)` to the sweeps at the end of the script. `make_results_figs.py` and `line_pressure_sweep.py` show how to override any module-level constant without editing the model.

## Limitations

The models are verified for internal consistency but not validated against measurements. Milk is treated as water, so predicted temperatures are lower bounds by up to about 5 °C. The jet-mixing constants (`C_jet`, `L_mix_D`) are uncalibrated, so the absolute mixing lengths are comparative rather than predictive. The denaturation kinetics (Safavi Nic et al., 2020) were fitted to a whey-enriched product and under-predict measured milk denaturation, so `denaturation.py` reports each injector relative to uniformly heated milk rather than as absolute values. The full list of assumptions (A1–A17) is in Appendix V of the report, and their consequences are discussed in Section 5.6.

## Licence

Released under the MIT licence; see `LICENSE`.
