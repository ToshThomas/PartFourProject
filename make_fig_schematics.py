"""Report schematics, drawn as plain engineering line drawings (not to scale).

  figs/fig_dsi_schematic.png   Figure 1.1  direct steam injection into a product pipe
  figs/fig_model_structure.png Figure 3.1  structure of the cross-sectional model
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                     "hatch.linewidth": 0.5})
K = "black"
FILL = "0.93"       # light grey for the highlighted layer / cell
PLUME = "0.82"      # grey for the condensing jet


def line(ax, x, y, lw=0.8, **kw):
    ax.plot(x, y, color=kw.pop("color", K), lw=lw, solid_capstyle="butt", **kw)


def arrow(ax, p0, p1, lw=0.8, head="-|>", ms=7):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle=head, lw=lw, color=K, shrinkA=0, shrinkB=0, mutation_scale=ms))


def label(ax, text, xy, xytext, ha="left", va="center"):
    """Plain text with a thin straight leader line, as on a drawing."""
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
                arrowprops=dict(arrowstyle="-", lw=0.5, color=K, shrinkA=1, shrinkB=0))


def hatched_wall(ax, x0, x1, y0, y1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="white", edgecolor=K, lw=0.8, hatch="////"))


def centreline(ax, x0, x1, y):
    line(ax, [x0, x1], [y, y], lw=0.5, ls=(0, (8, 2, 1.5, 2)))


def break_mark(ax, x, y0, y1, w=0.18):
    """Pair of zig-zag break lines across the pipe."""
    ys = np.linspace(y0, y1, 9)
    for dx in (0.0, 0.35):
        xs = x + dx + w*np.array([0, 1, -1, 1, -1, 1, -1, 1, 0])[:len(ys)]
        ax.fill_betweenx(ys, xs, xs + 0.35 if dx == 0 else xs, color="white", zorder=4)
        line(ax, xs, ys, lw=0.6, zorder=5)


def jet(xs, x_inj, top, pen, lam, grow):
    yc = top - pen*(1 - np.exp(-(xs - x_inj)/lam))
    return yc, 0.08 + grow*(xs - x_inj)


# =========================================================== Figure 1.1
def figure_1_1():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    r, t = 1.0, 0.22                      # pipe inner radius, wall thickness
    x0, x1, x_inj = 0.0, 19.0, 4.0
    # pipe walls (cut by the steam connection)
    hatched_wall(ax, x0, x_inj - 0.3, r, r + t); hatched_wall(ax, x_inj + 0.3, x1, r, r + t)
    hatched_wall(ax, x0, x1, -r - t, -r)
    centreline(ax, x0 - 0.3, x1 + 0.3, 0.0)
    # steam connection and nozzle
    hatched_wall(ax, x_inj - 0.52, x_inj - 0.3, r, r + 2.3); hatched_wall(ax, x_inj + 0.3, x_inj + 0.52, r, r + 2.3)
    line(ax, [x_inj - 0.3, x_inj - 0.12, x_inj + 0.12, x_inj + 0.3], [r, r - 0.25, r - 0.25, r])
    arrow(ax, (x_inj, r + 3.1), (x_inj, r + 1.6))
    ax.text(x_inj + 0.75, r + 2.75, "Saturated steam, 6 bar(a)", va="center")
    # condensing jet: plume that widens until it fills the pipe
    xs = np.linspace(x_inj, 6.6, 300)
    yc, b = jet(xs, x_inj, r - 0.25, 0.9, 0.7, 0.75)
    lo = np.clip(yc - b, -r, r)
    ax.fill_between(xs, lo, r, color=PLUME, lw=0, zorder=1)
    line(ax, xs, lo, lw=0.6)
    fade = LinearSegmentedColormap.from_list("fade", [PLUME, "white"])   # mixes out downstream
    ax.imshow(np.linspace(0, 1, 256)[None, :], extent=[6.6, 9.6, -r, r], cmap=fade, aspect="auto",
              interpolation="bicubic", zorder=1)
    # milk in, product out
    arrow(ax, (x0 - 1.8, 0.0), (x0 - 0.1, 0.0)); ax.text(x0 - 1.8, 0.3, "Milk in", va="bottom")
    # long holding tube indicated by a break
    break_mark(ax, 11.4, -r - t - 0.05, r + t + 0.05)
    # back-pressure valve and outlet
    vx = x1 + 0.9
    line(ax, [x1, vx - 0.45], [0, 0], lw=0.8); line(ax, [vx + 0.45, vx + 1.3], [0, 0], lw=0.8)
    ax.add_patch(Polygon([[vx - 0.45, 0.35], [vx - 0.45, -0.35], [vx + 0.45, 0.35], [vx + 0.45, -0.35]],
                         closed=True, facecolor="white", edgecolor=K, lw=0.8))
    line(ax, [vx, vx], [0, 0.7]); line(ax, [vx - 0.25, vx + 0.25], [0.7, 0.7])
    arrow(ax, (vx + 1.3, 0), (vx + 2.0, 0)); ax.text(vx + 2.1, 0, "To flash vessel", va="center")
    # labels
    label(ax, "Condensing jet", (5.0, 0.35), (5.6, -2.2))
    label(ax, "Back-pressure valve", (vx, -0.35), (vx - 0.8, -2.2))
    # holding-tube dimension
    yd = r + t + 0.9
    line(ax, [7.0, 7.0], [r + t + 0.15, yd + 0.25], lw=0.5); line(ax, [x1, x1], [r + t + 0.15, yd + 0.25], lw=0.5)
    arrow(ax, (7.0, yd), (x1, yd), lw=0.6, ms=6, head="<|-|>")
    ax.text(13.0, yd + 0.3, "Holding tube: ≥ 72 °C for ≥ 15 s", ha="center", va="bottom")
    ax.set_xlim(-2.0, x1 + 6.2); ax.set_ylim(-2.7, r + 3.4); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(os.path.join(HERE, "figs", "fig_dsi_schematic.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================== Figure 3.1
def figure_3_1():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1, 1.6]))
    N = 10                                   # layers drawn (the model uses 40)

    # ---- (a) cross-section
    R, t = 5.0, 0.5
    axa.add_patch(Wedge((0, 0), R + t, 0, 360, width=t, facecolor="white", edgecolor=K, lw=0.8, hatch="////"))
    j_lo, j_hi = 1.0, 2.0
    yy = np.linspace(j_lo, j_hi, 60); xx = np.sqrt(R**2 - yy**2)
    axa.add_patch(Polygon(np.c_[np.r_[xx, -xx[::-1]], np.r_[yy, yy[::-1]]], facecolor=FILL, edgecolor="none"))
    for k in range(1, N):
        y = -R + k*2*R/N; w = np.sqrt(R**2 - y**2)
        line(axa, [-w, w], [y, y], lw=0.5)
    a0, a1 = np.arcsin(j_lo/R), np.arcsin(j_hi/R); arc = np.linspace(a0, a1, 30)
    for s in (1, -1):
        line(axa, s*R*np.cos(arc), R*np.sin(arc), lw=2.4)
        am = 0.5*(a0 + a1)
        arrow(axa, (s*(R + t)*np.cos(am), (R + t)*np.sin(am)), (s*(R + t + 1.3)*np.cos(am), (R + t + 1.3)*np.sin(am)))
    for yf in (j_hi, j_lo):
        arrow(axa, (-3.2, yf - 0.5), (-3.2, yf + 0.5), head="<|-|>", ms=6)
    axa.add_patch(Rectangle((-0.3, R - 0.15), 0.6, t + 1.0, facecolor="white", edgecolor=K, lw=0.8, hatch="xxxx"))
    centreline(axa, -R - t - 0.6, R + t + 0.6, 0.0)
    axa.text(0.9, 1.5, "Layer $j$, area $A_j$", ha="center", va="center", fontsize=8.5)
    label(axa, "Injector", (0.3, R + 0.6), (1.8, R + 1.6))
    label(axa, "Wetted arc $P_j$", (-R*np.cos(a1), R*np.sin(a1)), (-6.9, 4.9))
    axa.text(-2.95, 2.35, "$q_{j+1/2}$", fontsize=8.5, va="center"); axa.text(-2.95, 0.6, "$q_{j-1/2}$", fontsize=8.5, va="center")
    axa.text(R + t + 1.0, 2.7, "$q_w$", fontsize=8.5)
    axa.text(-6.9, -6.4, "(a)", fontsize=9)
    axa.set_xlim(-7.2, 7.6); axa.set_ylim(-6.6, 7.2); axa.set_aspect("equal"); axa.axis("off")

    # ---- (b) longitudinal section
    L, D, tw = 12.0, 10.0, 0.5
    hatched_wall(axb, 0, L, D, D + tw); hatched_wall(axb, 0, L, -tw, 0)
    for k in range(1, N):
        line(axb, [0, L], [k*D/N, k*D/N], lw=0.5)
    # inlet velocity profile with envelope
    yk = (np.arange(N) + 0.5)*D/N
    u = (np.minimum(yk, D - yk)/(D/2))**(1/7)*1.4
    for y_, u_ in zip(yk, u):
        arrow(axb, (0.25, y_), (0.25 + u_, y_), lw=0.6, ms=5)
    ye = np.linspace(0.02, D - 0.02, 200)
    line(axb, 0.25 + 1.4*(np.minimum(ye, D - ye)/(D/2))**(1/7), ye, lw=0.6)
    line(axb, [0.25, 0.25], [0, D], lw=0.6)
    # injector, jet and plume
    x_inj = 3.2
    axb.add_patch(Rectangle((x_inj - 0.3, D - 0.1), 0.6, tw + 1.0, facecolor="white", edgecolor=K, lw=0.8, hatch="xxxx"))
    xs = np.linspace(x_inj, L, 400)
    yc, b = jet(xs, x_inj, D, 4.0, 0.9, 1.6)
    lo = np.clip(yc - b, 0, D)
    axb.fill_between(xs, lo, D, color=PLUME, lw=0, zorder=0.5)
    m = xs < 6.6
    line(axb, xs[m], lo[m], lw=0.6)
    line(axb, xs[xs < 6.4], yc[xs < 6.4], lw=0.6, ls=(0, (4, 2)))
    # one cell and its balance
    cx, cw, cy, ch = 8.0, 0.9, 6.0, 1.0
    axb.add_patch(Rectangle((cx, cy), cw, ch, facecolor="white", edgecolor=K, lw=1.1, zorder=3))
    arrow(axb, (cx - 1.2, cy + ch/2), (cx, cy + ch/2))
    arrow(axb, (cx + cw, cy + ch/2), (cx + cw + 1.2, cy + ch/2))
    wb = dict(facecolor="white", edgecolor="none", pad=0.5)
    axb.text(cx - 1.5, cy + ch/2 + 0.3, "$H_j(x)$", fontsize=8.5, va="bottom", bbox=wb)
    axb.text(cx + cw + 0.1, cy + ch/2 + 0.3, r"$H_j(x+\Delta x)$", fontsize=8.5, va="bottom", bbox=wb)
    for yf in (cy + ch, cy):
        arrow(axb, (cx + cw/2, yf - 0.4), (cx + cw/2, yf + 0.4), head="<|-|>", ms=5)
    axb.text(cx + cw/2, cy + ch/2, "$S_j$", fontsize=8.5, ha="center", va="center")
    line(axb, [cx, cx], [cy - 0.25, cy - 1.0], lw=0.5); line(axb, [cx + cw, cx + cw], [cy - 0.25, cy - 1.0], lw=0.5)
    arrow(axb, (cx + cw/2, cy - 0.8), (cx, cy - 0.8), lw=0.5, ms=5); arrow(axb, (cx + cw/2, cy - 0.8), (cx + cw, cy - 0.8), lw=0.5, ms=5)
    axb.text(cx + cw/2, cy - 1.0, r"$\Delta x$", fontsize=8.5, ha="center", va="top", bbox=wb)
    # wall loss
    for xw in (9.0, 10.3, 11.6):
        arrow(axb, (xw, D + tw), (xw, D + tw + 1.0), ms=6); arrow(axb, (xw, -tw), (xw, -tw - 1.0), ms=6)
    axb.text(10.3, D + tw + 1.15, "$q_w$", fontsize=8.5, ha="center", va="bottom")
    label(axb, "Injector", (x_inj + 0.3, D + 1.0), (x_inj + 1.0, D + 1.9))
    label(axb, "Condensing jet", (5.2, 6.0), (5.6, 2.0))
    # x axis
    arrow(axb, (0, -2.4), (2.2, -2.4)); axb.text(2.35, -2.4, "$x$", va="center", style="italic")
    axb.text(-0.4, -3.3, "(b)", fontsize=9)
    axb.set_xlim(-0.5, L + 0.4); axb.set_ylim(-3.6, D + 2.6); axb.set_aspect("equal"); axb.axis("off")

    plt.tight_layout(w_pad=0.5)
    fig.savefig(os.path.join(HERE, "figs", "fig_model_structure.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    figure_1_1()
    figure_3_1()
    print("wrote figs/fig_dsi_schematic.png and figs/fig_model_structure.png")
