#!/usr/bin/env python3
"""
MECH.4730 - Solid vs. Hollow Shaft Equivalence: Cost-Savings Optimizer (GUI)
=============================================================================

Finds the hollow steel shaft (Do, Di) that MAXIMIZES COST SAVINGS while being
equivalent-or-better than the existing 3.0 in solid shaft, then shows every other
feasible design for comparison.

Method
  1. Vectorized grid sweep of every (Do, Di) at STEP resolution -> feasible set.
  2. SLSQP constrained optimizer (scipy), seeded with the best grid point, to find
     the exact continuous optimum. Verified against all constraints afterwards.

Objective
  maximize savings = 1 - premium * (Do^2 - Di^2) / D^2   (same as minimizing area)

Constraints (all user-editable in the GUI)
  Do <= tunnel - clearance                 envelope
  A/A0 <= max weight ratio                 weight
  savings >= min savings                   cost
  Z/Z0 >= min strength ratio               bending & torsional stress (J = 2I)
  I/I0 >= 1                                load deflection & twist
  sag <= sag0                              self-weight deflection
  Nc >= margin * N_op                      first critical speed (exact; Rayleigh optional)
  Do/t <= limit                            thin wall / local buckling

Equations
  A = pi/4 (Do^2 - Di^2)          I = pi/64 (Do^4 - Di^4)       J = 2I
  Z = I / (Do/2)                  Zp = J / (Do/2)               G = E / (2(1+nu))
  sag = 5 w L^4 / (384 E I),  w = rho A
  wn = (pi/L)^2 sqrt(E I g / (rho A))   (exact, uniform simply supported shaft)
  wn = sqrt(g / sag)                    (Rayleigh, conservative)
  Nc = 30 wn / pi  [RPM]

Usage
  python shaft_optimizer.py            -> GUI window
  python shaft_optimizer.py --nogui    -> console output + files in ./results
"""

import csv
import math
import os
import sys
from dataclasses import dataclass, fields, asdict

import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except ImportError:  # grid result is still exact to STEP
    HAVE_SCIPY = False

# Chart colours (validated categorical palette, light mode)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ==================================================================================
# INPUTS
# ==================================================================================
@dataclass
class Inputs:
    D_solid: float = 3.0          # in
    L: float = 90.0               # in, bearing span
    E: float = 30e6               # psi
    nu: float = 0.29              # Poisson's ratio (for G only)
    rho: float = 0.28             # lb/in^3
    g: float = 386.1              # in/s^2
    N_op: float = 2000.0          # RPM
    tunnel_D: float = 4.0         # in, hard limit
    clearance: float = 0.05       # in, diametral clearance inside tunnel
    max_weight_ratio: float = 0.50
    min_savings: float = 0.30
    cost_premium: float = 1.20    # hollow $/lb relative to solid
    crit_margin: float = 1.25     # Nc >= margin * N_op
    max_do_over_t: float = 20.0
    min_z_ratio: float = 1.00     # strength margin vs solid (1.0 = equal)
    require_rayleigh: bool = False
    step: float = 0.005           # in, grid resolution

    @property
    def Do_max(self):
        return self.tunnel_D - self.clearance

    @property
    def G(self):
        return self.E / (2 * (1 + self.nu))


# (field, label, unit, format) shown in the GUI input panel
INPUT_FIELDS = [
    ("D_solid", "Solid shaft diameter", "in"),
    ("L", "Bearing span", "in"),
    ("E", "Young's modulus E", "psi"),
    ("nu", "Poisson's ratio", "-"),
    ("rho", "Density", "lb/in³"),
    ("g", "Gravity", "in/s²"),
    ("N_op", "Operating speed", "RPM"),
    ("tunnel_D", "Tunnel diameter (hard)", "in"),
    ("clearance", "Diametral clearance", "in"),
    ("max_weight_ratio", "Max weight ratio", "×solid"),
    ("min_savings", "Min cost savings", "fraction"),
    ("cost_premium", "Hollow $/lb premium", "×solid"),
    ("crit_margin", "Critical speed margin", "×Nop"),
    ("max_do_over_t", "Max Do/t", "-"),
    ("min_z_ratio", "Min strength ratio Z/Z₀", "-"),
    ("step", "Grid step", "in"),
]


# ==================================================================================
# PHYSICS (vectorized)
# ==================================================================================
def props(do, di, p: Inputs):
    do = np.asarray(do, dtype=float)
    di = np.asarray(di, dtype=float)
    A = np.pi / 4 * (do**2 - di**2)
    I = np.pi / 64 * (do**4 - di**4)
    J = 2 * I
    Z = I / (do / 2)
    w = p.rho * A
    sag = 5 * w * p.L**4 / (384 * p.E * I)
    wn = (np.pi / p.L) ** 2 * np.sqrt(p.E * I * p.g / (p.rho * A))
    wr = np.sqrt(p.g / sag)
    return dict(A=A, I=I, J=J, Z=Z, weight=w * p.L, sag=sag,
                Nc=wn * 30 / np.pi, Nc_ray=wr * 30 / np.pi)


def evaluate(do, di, p: Inputs, solid=None):
    """Full result set (arrays) for any number of designs."""
    s0 = solid or {k: float(v) for k, v in props(p.D_solid, 0.0, p).items()}
    s = props(do, di, p)
    do = np.asarray(do, dtype=float)
    di = np.asarray(di, dtype=float)
    t = (do - di) / 2
    wr = s["A"] / s0["A"]
    r = dict(
        Do=do, Di=di, t=t, Do_over_t=do / t,
        weight_lb=s["weight"], weight_ratio=wr, weight_reduction=1 - wr,
        cost_ratio=p.cost_premium * wr, cost_savings=1 - p.cost_premium * wr,
        Z_ratio=s["Z"] / s0["Z"], I_ratio=s["I"] / s0["I"],
        stress_ratio=s0["Z"] / s["Z"],
        twist_ratio=s0["J"] / s["J"],
        sag_in=s["sag"], sag_ratio=s["sag"] / s0["sag"],
        Nc_rpm=s["Nc"], Nc_ray_rpm=s["Nc_ray"],
        Nc_over_Nop=s["Nc"] / p.N_op, Nc_ray_over_Nop=s["Nc_ray"] / p.N_op,
    )
    eps = 1e-9
    req_nc = p.crit_margin * p.N_op
    ok = dict(
        envelope=do <= p.Do_max + eps,
        weight=wr <= p.max_weight_ratio + eps,
        cost=r["cost_savings"] >= p.min_savings - eps,
        strength=r["Z_ratio"] >= p.min_z_ratio - eps,
        stiffness=r["I_ratio"] >= 1 - eps,
        sag=r["sag_ratio"] <= 1 + eps,
        crit_speed=(s["Nc"] >= req_nc - eps) & ((s["Nc_ray"] >= req_nc - eps)
                                               if p.require_rayleigh else True),
        wall=r["Do_over_t"] <= p.max_do_over_t + eps,
    )
    feasible = np.ones_like(do, dtype=bool)
    for v in ok.values():
        feasible &= v
    r["feasible"] = feasible
    r["_ok"] = ok
    return r


def solid_baseline(p: Inputs):
    s = {k: float(v) for k, v in props(p.D_solid, 0.0, p).items()}
    r = row(evaluate(p.D_solid, 0.0, p, s), None)
    r.update(cost_ratio=1.0, cost_savings=0.0, Do_over_t=p.D_solid / (p.D_solid / 2))
    r["raw"] = s
    return r


def row(res, idx):
    """Extract one design as a plain dict of floats."""
    out = {}
    for k, v in res.items():
        if k.startswith("_"):
            continue
        a = np.asarray(v)
        out[k] = a.item() if idx is None else a[idx].item()
    shape = np.shape(res["Do"])
    fails = []
    for k, v in res["_ok"].items():
        a = np.broadcast_to(np.asarray(v), shape)
        ok = bool(a.item() if idx is None else a[idx])
        if not ok:
            fails.append(k)
    out["fails"] = ", ".join(fails)
    return out


# ==================================================================================
# SOLVER
# ==================================================================================
def sweep(p: Inputs):
    do_min = math.sqrt(p.max_weight_ratio) * p.D_solid
    start = math.ceil(do_min / p.step) * p.step
    do_vals = np.round(np.arange(start, p.Do_max + p.step / 2, p.step), 6)
    di_vals = np.round(np.arange(0.0, p.Do_max, p.step), 6)
    DO, DI = np.meshgrid(do_vals, di_vals, indexing="ij")
    mask = DI <= DO - 2 * p.step
    do, di = DO[mask], DI[mask]
    res = evaluate(do, di, p)
    return res


def optimize(p: Inputs, seed):
    """SLSQP: maximize savings subject to all constraints. Returns (do, di, success)."""
    if not HAVE_SCIPY:
        return seed[0], seed[1], False
    s0 = {k: float(v) for k, v in props(p.D_solid, 0.0, p).items()}
    req = p.crit_margin * p.N_op

    def f(x):
        r = props(x[0], x[1], p)
        return float(p.cost_premium * r["A"] / s0["A"])   # cost ratio -> minimize

    def g(x):
        do, di = x
        r = props(do, di, p)
        t = (do - di) / 2
        c = [
            p.Do_max - do,
            p.max_weight_ratio - r["A"] / s0["A"],
            (1 - p.cost_premium * r["A"] / s0["A"]) - p.min_savings,
            r["Z"] / s0["Z"] - p.min_z_ratio,
            r["I"] / s0["I"] - 1,
            1 - r["sag"] / s0["sag"],
            r["Nc"] / req - 1,
            p.max_do_over_t - do / max(t, 1e-9),
            do - di - 0.02,
            di,
        ]
        if p.require_rayleigh:
            c.append(r["Nc_ray"] / req - 1)
        return np.array([float(v) for v in c])

    sol = minimize(f, x0=np.array(seed, dtype=float), method="SLSQP",
                   constraints=[{"type": "ineq", "fun": g}],
                   bounds=[(0.1, p.Do_max), (0.0, p.Do_max - 0.02)],
                   options=dict(ftol=1e-12, maxiter=500))
    do, di = (float(v) for v in sol.x)
    # Back off Di microscopically if numerical slack violates a constraint
    for _ in range(60):
        if bool(evaluate(do, di, p)["feasible"]):
            return do, di, bool(sol.success)
        di -= 1e-6
    return seed[0], seed[1], False


def solve(p: Inputs):
    res = sweep(p)
    feas_idx = np.flatnonzero(res["feasible"])
    out = dict(inputs=p, solid=solid_baseline(p), n_grid=int(res["Do"].size),
               n_feasible=int(feas_idx.size), scipy=HAVE_SCIPY)
    if feas_idx.size == 0:
        # report the binding failures of the closest designs
        fails = {}
        for k, v in res["_ok"].items():
            fails[k] = int(np.size(v) - np.count_nonzero(v)) if not isinstance(v, bool) else 0
        out.update(feasible=None, optimum=None, alternatives={}, fail_counts=fails)
        return out

    feas = {k: (np.asarray(v)[feas_idx] if not k.startswith("_") else None)
            for k, v in res.items()}
    feas.pop("_ok")
    order = np.argsort(-feas["cost_savings"])
    feas = {k: v[order] for k, v in feas.items()}

    best_grid = row(evaluate(feas["Do"][0], feas["Di"][0], p), None)
    do, di, ok = optimize(p, (best_grid["Do"], best_grid["Di"]))
    opt = row(evaluate(do, di, p), None)
    if opt["cost_savings"] < best_grid["cost_savings"]:
        opt, ok = best_grid, False
    opt["method"] = "SLSQP (continuous)" if ok else "grid search"

    # Reference alternatives
    alts = {}
    # A: exactly max weight ratio, Z ratio = min (smallest Do)
    a_do = _bisect(lambda d: _di_strength(d, p) - _di_weight(d, p),
                   math.sqrt(p.max_weight_ratio) * p.D_solid + 1e-6, p.Do_max)
    if a_do:
        alts["A: smallest Do at weight limit"] = row(
            evaluate(a_do, _di_weight(a_do, p), p), None)
    # B: weight at limit, largest Do -> max margins
    b = row(evaluate(p.Do_max, _di_weight(p.Do_max, p), p), None)
    if b["feasible"]:
        alts["B: max margin at weight limit"] = b
    alts["OPTIMUM: max cost savings"] = opt

    out.update(feasible=feas, optimum=opt, alternatives=alts, best_grid=best_grid)
    return out


def _di_weight(do, p, ratio=None):
    v = do**2 - (ratio or p.max_weight_ratio) * p.D_solid**2
    return math.sqrt(v) if v > 0 else 0.0


def _di_strength(do, p):
    v = do**4 - p.min_z_ratio * p.D_solid**3 * do
    return v**0.25 if v > 0 else 0.0


def _di_crit(do, p):
    k = (math.pi / p.L) ** 2 * math.sqrt(p.E * p.g / p.rho)
    wn = p.crit_margin * p.N_op * math.pi / 30
    v = 16 * (wn / k) ** 2 - do**2
    return math.sqrt(v) if v > 0 else 0.0


def _di_wall(do, p):
    return do - 2 * do / p.max_do_over_t


def _bisect(f, lo, hi, n=200):
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(n):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ==================================================================================
# REPORT TEXT / FILES
# ==================================================================================
TABLE_COLS = [
    ("Do", "Do (in)", "{:.4f}"), ("Di", "Di (in)", "{:.4f}"), ("t", "Wall t (in)", "{:.4f}"),
    ("Do_over_t", "Do/t", "{:.1f}"), ("weight_lb", "Weight (lb)", "{:.1f}"),
    ("weight_reduction", "Wt reduction", "{:.1%}"), ("cost_savings", "Cost savings", "{:.2%}"),
    ("Z_ratio", "Z/Z₀", "{:.3f}"), ("stress_ratio", "σ/σ₀", "{:.3f}"),
    ("I_ratio", "I/I₀", "{:.3f}"), ("sag_ratio", "Sag/sag₀", "{:.3f}"),
    ("Nc_rpm", "Nc exact (RPM)", "{:.0f}"), ("Nc_ray_rpm", "Nc Rayleigh (RPM)", "{:.0f}"),
    ("Nc_over_Nop", "Nc/Nop", "{:.2f}"),
]


def solid_verdict(sol):
    p, s = sol["inputs"], sol["solid"]
    ratio = p.N_op / s["Nc_rpm"]
    band = p.crit_margin - 1
    if s["Nc_rpm"] >= p.crit_margin * p.N_op:
        return f"SAFE: first critical {s['Nc_rpm']:.0f} RPM is ≥ {p.crit_margin}× operating speed"
    if abs(ratio - 1) < band:
        side = "ABOVE" if ratio > 1 else "BELOW"
        return (f"NOT SAFE: operating {ratio:.2f}× its first critical speed "
                f"({s['Nc_rpm']:.0f} RPM), {side} critical and inside the ±{band:.0%} band")
    return f"Operating {ratio:.2f}× first critical ({s['Nc_rpm']:.0f} RPM); check margin"


def summary_text(sol):
    p, s = sol["inputs"], sol["solid"]
    L = []
    L.append("MECH.4730  SOLID vs HOLLOW SHAFT  —  COST-SAVINGS OPTIMIZATION")
    L.append("=" * 72)
    L.append(f"Existing solid shaft: D = {p.D_solid:.3f} in, weight {s['weight_lb']:.1f} lb")
    L.append(f"  Nc exact {s['Nc_rpm']:.0f} RPM | Rayleigh {s['Nc_ray_rpm']:.0f} RPM")
    L.append(f"  At {p.N_op:.0f} RPM: {solid_verdict(sol)}")
    L.append("")
    L.append(f"Grid designs evaluated: {sol['n_grid']:,} (step {p.step} in)")
    L.append(f"Feasible designs:       {sol['n_feasible']:,}")
    if not sol["optimum"]:
        L.append("\nNO FEASIBLE DESIGN. Designs rejected per constraint:")
        for k, v in sol["fail_counts"].items():
            L.append(f"  {k:<11} {v:,}")
        return "\n".join(L)
    f = sol["feasible"]
    L.append(f"  Do range       {f['Do'].min():.3f} – {f['Do'].max():.3f} in (max {p.Do_max:.3f})")
    L.append(f"  Weight red.    {f['weight_reduction'].min():.1%} – {f['weight_reduction'].max():.1%}")
    L.append(f"  Cost savings   {f['cost_savings'].min():.1%} – {f['cost_savings'].max():.1%}")
    L.append(f"  Nc exact       {f['Nc_rpm'].min():.0f} – {f['Nc_rpm'].max():.0f} RPM")
    o = sol["optimum"]
    L.append("")
    L.append(f"OPTIMUM (max cost savings) — {o['method']}")
    L.append("-" * 72)
    L.append(f"  Do = {o['Do']:.4f} in   Di = {o['Di']:.4f} in   wall t = {o['t']:.4f} in"
             f"   Do/t = {o['Do_over_t']:.1f}")
    L.append(f"  Weight {o['weight_lb']:.1f} lb  ({o['weight_reduction']:.1%} lighter)")
    L.append(f"  COST SAVINGS = {o['cost_savings']:.2%}")
    L.append(f"  Strength Z/Z₀ = {o['Z_ratio']:.3f}  (stress {o['stress_ratio']:.3f}× solid)")
    L.append(f"  Stiffness I/I₀ = {o['I_ratio']:.3f}  Self-weight sag {o['sag_ratio']:.3f}× solid")
    L.append(f"  Nc exact {o['Nc_rpm']:.0f} RPM ({o['Nc_over_Nop']:.2f}× Nop)"
             f" | Rayleigh {o['Nc_ray_rpm']:.0f} RPM ({o['Nc_ray_over_Nop']:.2f}× Nop)")
    L.append(f"  At {p.N_op:.0f} RPM: SAFE (critical ≥ {p.crit_margin}× operating)")
    L.append("")
    L.append("Binding constraints at optimum: " + ", ".join(binding(o, p)))
    L.append("Tip: raise 'Min strength ratio' (e.g. 1.05) to buy margin at small cost.")
    if not sol["scipy"]:
        L.append("\n(scipy not installed: optimum is the best grid point, accurate to the grid step)")
    return "\n".join(L)


def binding(o, p, tol=2e-3):
    b = []
    if p.Do_max - o["Do"] < tol:
        b.append(f"envelope (Do = {p.Do_max:.3f})")
    if o["Z_ratio"] - p.min_z_ratio < tol:
        b.append(f"strength (Z/Z₀ = {p.min_z_ratio:.2f})")
    if p.max_weight_ratio - o["weight_ratio"] < tol:
        b.append("weight")
    if o["Nc_rpm"] / (p.crit_margin * p.N_op) - 1 < tol:
        b.append("critical speed")
    if p.max_do_over_t - o["Do_over_t"] < 0.05:
        b.append("Do/t")
    return b or ["none"]


def write_outputs(sol, folder, figs=None):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary_text(sol))
        fh.write("\n\nINPUTS\n")
        for k, v in asdict(sol["inputs"]).items():
            fh.write(f"  {k} = {v}\n")
    if sol["feasible"]:
        f = sol["feasible"]
        keys = [k for k in f.keys() if k != "feasible"]
        with open(os.path.join(folder, "feasible_designs.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for i in range(len(f["Do"])):
                w.writerow([round(float(f[k][i]), 6) for k in keys])
        rows = [("Solid", sol["solid"])] + list(sol["alternatives"].items())
        with open(os.path.join(folder, "selected_designs.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["design"] + [c[1] for c in TABLE_COLS])
            for name, r in rows:
                w.writerow([name] + [c[2].format(r[c[0]]) for c in TABLE_COLS])
    for name, fig in (figs or {}).items():
        fig.savefig(os.path.join(folder, f"{name}.png"), dpi=150)


# ==================================================================================
# PLOTS (matplotlib Figure objects; no pyplot so they embed cleanly in Tk)
# ==================================================================================
def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(sol):
    from matplotlib.figure import Figure
    p = sol["inputs"]
    figs = {}
    f = sol["feasible"]
    o = sol["optimum"]

    # 1. Design space
    fig = Figure(figsize=(9, 6), dpi=100, facecolor="white")
    ax = fig.add_subplot(111)
    _style(ax)
    if f:
        ax.scatter(f["Do"], f["Di"], s=4, color=C_AQUA, alpha=0.4, linewidths=0,
                   label="Feasible designs")
        x0, x1 = f["Do"].min() - 0.15, p.tunnel_D + 0.05
        y0, y1 = f["Di"].min() - 0.25, f["Di"].max() + 0.3
    else:
        x0, x1, y0, y1 = 1.5, p.tunnel_D + 0.1, 0, p.tunnel_D
    do = np.linspace(max(0.5, x0), x1, 400)
    curves = [
        ([_di_weight(d, p) for d in do], C_BLUE, f"Weight = {p.max_weight_ratio:.0%} of solid (Di ≥)"),
        ([_di_strength(d, p) for d in do], C_ORANGE, f"Strength Z/Z₀ = {p.min_z_ratio:.2f} (Di ≤)"),
        ([_di_crit(d, p) for d in do], C_VIOLET, f"Nc = {p.crit_margin:.2f}×{p.N_op:.0f} RPM (Di ≥)"),
        ([_di_wall(d, p) for d in do], C_MAGENTA, f"Do/t = {p.max_do_over_t:.0f} (Di ≤)"),
    ]
    for y, c, lab in curves:
        ax.plot(do, y, color=c, linewidth=2, label=lab)
    ax.axvline(p.Do_max, color=INK2, linewidth=1.5, linestyle="--",
               label=f"Max Do = {p.Do_max:.2f} in")
    ax.axvline(p.tunnel_D, color=C_RED, linewidth=1.5, label=f"Tunnel wall {p.tunnel_D:.2f} in")
    if o:
        marks = {"A": "o", "B": "s", "OPTIMUM": "*"}
        for name, r in sol["alternatives"].items():
            key = name.split(":")[0]
            star = key == "OPTIMUM"
            ax.scatter([r["Do"]], [r["Di"]], s=260 if star else 70, marker=marks.get(key, "o"),
                       color=C_YELLOW if star else INK, edgecolors=INK if star else SURF,
                       linewidths=1.2 if star else 2, zorder=6)
            ax.annotate(name, (r["Do"], r["Di"]),
                        xytext=(-12, 12) if star else (10, -14), textcoords="offset points",
                        fontsize=8.5, color=INK, ha="right" if star else "left",
                        fontweight="bold" if star else "normal")
    ax.set_xlim(x0, x1)
    ax.set_ylim(max(0, y0), y1)
    ax.set_xlabel("Outside diameter Do (in)", color=INK)
    ax.set_ylabel("Inside diameter Di (in)", color=INK)
    ax.set_title("Design space (feasible region shaded, ★ = max cost savings)",
                 color=INK, fontsize=12, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    figs["design_space"] = fig

    # 2. Critical speed along the weight-limit line
    fig = Figure(figsize=(9, 5), dpi=100, facecolor="white")
    ax = fig.add_subplot(111)
    _style(ax)
    lo = math.sqrt(p.max_weight_ratio) * p.D_solid + 0.01
    do = np.linspace(lo, p.Do_max, 300)
    di = np.array([_di_weight(d, p) for d in do])
    pr = props(do, di, p)
    s = sol["solid"]
    ax.plot(do, pr["Nc"], color=C_BLUE, linewidth=2, label="Hollow, exact")
    ax.plot(do, pr["Nc_ray"], color=C_ORANGE, linewidth=2, label="Hollow, Rayleigh")
    ax.axhline(s["Nc_rpm"], color=C_BLUE, linestyle=":", linewidth=1.5,
               label=f"Solid, exact {s['Nc_rpm']:.0f} RPM")
    ax.axhline(s["Nc_ray_rpm"], color=C_ORANGE, linestyle=":", linewidth=1.5,
               label=f"Solid, Rayleigh {s['Nc_ray_rpm']:.0f} RPM")
    ax.axhline(p.N_op, color=C_RED, linewidth=2, label=f"Operating {p.N_op:.0f} RPM")
    ax.axhline(p.crit_margin * p.N_op, color=INK2, linestyle="--", linewidth=1.5,
               label=f"Required ≥ {p.crit_margin * p.N_op:.0f} RPM")
    if o:
        ax.scatter([o["Do"]], [o["Nc_rpm"]], s=220, marker="*", color=C_YELLOW,
                   edgecolors=INK, linewidths=1.2, zorder=6,
                   label=f"Optimum {o['Nc_rpm']:.0f} RPM")
    ax.set_xlabel(f"Outside diameter Do (in), weight held at {p.max_weight_ratio:.0%}", color=INK)
    ax.set_ylabel("First critical speed (RPM)", color=INK)
    ax.set_title("First critical speed: hollow vs. solid", color=INK, fontsize=12, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    figs["critical_speed"] = fig

    # 3. Savings map
    fig = Figure(figsize=(9, 6), dpi=100, facecolor="white")
    ax = fig.add_subplot(111)
    _style(ax)
    if f:
        sc = ax.scatter(f["Do"], f["Di"], s=10, c=100 * f["cost_savings"], cmap="Blues",
                        linewidths=0)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Cost savings vs. solid (%)", color=INK)
        cb.ax.tick_params(colors=INK2, labelsize=9)
        ax.scatter([o["Do"]], [o["Di"]], s=260, marker="*", color=C_YELLOW, edgecolors=INK,
                   linewidths=1.2, zorder=6, label=f"Optimum: {o['cost_savings']:.2%} savings")
        ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_xlabel("Outside diameter Do (in)", color=INK)
    ax.set_ylabel("Inside diameter Di (in)", color=INK)
    ax.set_title("Cost savings across feasible designs", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    figs["savings_map"] = fig
    return figs


# ==================================================================================
# GUI (tkinter — ships with the python.org Windows installer)
# ==================================================================================
def fmt_val(v):
    return f"{v:.0f}" if abs(v) >= 1e5 else f"{v:g}"


def run_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    if sys.platform == "win32":  # crisp text on high-DPI Windows displays
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    root = tk.Tk()
    root.title("MECH.4730 — Hollow Shaft Cost-Savings Optimizer")
    root.geometry("1400x860")
    root.minsize(1100, 700)
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    style.configure("Treeview", rowheight=22)
    style.configure("Big.TButton", font=("Segoe UI", 10, "bold"))

    state = {"sol": None, "figs": {}, "canvases": []}

    # ---------- left: inputs ----------
    left = ttk.Frame(root, padding=10)
    left.pack(side="left", fill="y")
    ttk.Label(left, text="Inputs & constraints", font=("Segoe UI", 11, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
    defaults = Inputs()
    vars_ = {}
    for i, (key, label, unit) in enumerate(INPUT_FIELDS, start=1):
        ttk.Label(left, text=label).grid(row=i, column=0, sticky="w", pady=2)
        v = tk.StringVar(value=fmt_val(getattr(defaults, key)))
        ttk.Entry(left, textvariable=v, width=11, justify="right").grid(row=i, column=1, padx=4)
        ttk.Label(left, text=unit, foreground=INK2).grid(row=i, column=2, sticky="w")
        vars_[key] = v
    r = len(INPUT_FIELDS) + 1
    ray_var = tk.BooleanVar(value=defaults.require_rayleigh)
    ttk.Checkbutton(left, text="Rayleigh critical speed must also pass",
                    variable=ray_var).grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 2))
    r += 1
    ttk.Button(left, text="▶  Optimize", style="Big.TButton",
               command=lambda: run()).grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 4))
    ttk.Button(left, text="Reset defaults", command=lambda: reset()).grid(
        row=r + 1, column=0, columnspan=3, sticky="ew", pady=2)
    ttk.Button(left, text="Export results…", command=lambda: export()).grid(
        row=r + 2, column=0, columnspan=3, sticky="ew", pady=2)
    status = tk.StringVar(value="Ready.")
    ttk.Label(left, textvariable=status, foreground=INK2, wraplength=260).grid(
        row=r + 3, column=0, columnspan=3, sticky="w", pady=(10, 0))
    if not HAVE_SCIPY:
        ttk.Label(left, text="scipy not found: using grid optimum only",
                  foreground=C_RED, wraplength=260).grid(row=r + 4, column=0, columnspan=3,
                                                         sticky="w")

    # ---------- right: tabs ----------
    nb = ttk.Notebook(root)
    nb.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=10)

    tab_sum = ttk.Frame(nb, padding=8)
    nb.add(tab_sum, text="Summary")
    txt = tk.Text(tab_sum, wrap="none", font=("Consolas", 10), relief="flat",
                  background=SURF, foreground=INK)
    txt.pack(side="top", fill="both", expand=True)

    ttk.Label(tab_sum, text="Solid baseline vs. selected hollow designs",
              font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 2))
    cols = ["Design"] + [c[1] for c in TABLE_COLS]
    sel_tree = ttk.Treeview(tab_sum, columns=cols, show="headings", height=5)
    for c in cols:
        sel_tree.heading(c, text=c)
        sel_tree.column(c, width=240 if c == "Design" else 92, anchor="e" if c != "Design" else "w")
    sel_tree.pack(fill="x")
    xs = ttk.Scrollbar(tab_sum, orient="horizontal", command=sel_tree.xview)
    sel_tree.configure(xscrollcommand=xs.set)
    xs.pack(fill="x")

    tab_tbl = ttk.Frame(nb, padding=8)
    nb.add(tab_tbl, text="All feasible designs")
    ttk.Label(tab_tbl, text="Sorted by cost savings (click a header to re-sort)",
              foreground=INK2).pack(anchor="w")
    tbl_cols = [c[1] for c in TABLE_COLS]
    tree = ttk.Treeview(tab_tbl, columns=tbl_cols, show="headings")
    vs = ttk.Scrollbar(tab_tbl, orient="vertical", command=tree.yview)
    hs = ttk.Scrollbar(tab_tbl, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
    vs.pack(side="right", fill="y")
    hs.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)
    sort_dir = {}

    def sort_by(ci):
        f = state["sol"] and state["sol"]["feasible"]
        if not f:
            return
        key = TABLE_COLS[ci][0]
        desc = not sort_dir.get(key, key == "cost_savings")
        sort_dir[key] = desc
        fill_tree(np.argsort(-f[key] if desc else f[key], kind="stable"))

    for i, c in enumerate(tbl_cols):
        tree.heading(c, text=c, command=lambda i=i: sort_by(i))
        tree.column(c, width=100, anchor="e")

    plot_tabs = {}
    for key, title in (("design_space", "Design space"), ("critical_speed", "Critical speed"),
                       ("savings_map", "Savings map")):
        fr = ttk.Frame(nb)
        nb.add(fr, text=title)
        plot_tabs[key] = fr

    # ---------- actions ----------
    def read_inputs():
        kw = {}
        for key, label, _ in INPUT_FIELDS:
            try:
                kw[key] = float(vars_[key].get())
            except ValueError:
                raise ValueError(f"'{label}' must be a number.")
        kw["require_rayleigh"] = bool(ray_var.get())
        p = Inputs(**kw)
        if p.step <= 0 or p.step > 0.1:
            raise ValueError("Grid step must be between 0 and 0.1 in.")
        if p.Do_max <= 0:
            raise ValueError("Tunnel diameter minus clearance must be positive.")
        n = ((p.Do_max - math.sqrt(p.max_weight_ratio) * p.D_solid) / p.step) * (p.Do_max / p.step) / 2
        if n > 6e6:
            raise ValueError("Grid too fine (> 6 million designs). Increase the grid step.")
        return p

    def fill_tree(order=None):
        tree.delete(*tree.get_children())
        f = state["sol"]["feasible"]
        if not f:
            return
        idx = range(len(f["Do"])) if order is None else order
        opt = state["sol"]["optimum"]
        for i in idx:
            vals = [c[2].format(float(f[c[0]][i])) for c in TABLE_COLS]
            tree.insert("", "end", values=vals)
        tree.insert("", 0, values=[c[2].format(opt[c[0]]) for c in TABLE_COLS], tags=("opt",))
        tree.tag_configure("opt", background="#fde9b5")

    def show_plots():
        for c in state["canvases"]:
            c.get_tk_widget().destroy()
        for fr in plot_tabs.values():
            for w in fr.winfo_children():
                w.destroy()
        state["canvases"] = []
        for key, fig in state["figs"].items():
            fr = plot_tabs[key]
            canvas = FigureCanvasTkAgg(fig, master=fr)
            tb = NavigationToolbar2Tk(canvas, fr, pack_toolbar=False)
            tb.update()
            tb.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()
            state["canvases"].append(canvas)

    def run():
        try:
            p = read_inputs()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        status.set("Optimizing…")
        root.config(cursor="watch")
        root.update_idletasks()
        try:
            sol = solve(p)
            state["sol"] = sol
            state["figs"] = make_figures(sol)
        except Exception as e:  # noqa: BLE001
            root.config(cursor="")
            messagebox.showerror("Solver error", repr(e))
            status.set("Error.")
            return
        root.config(cursor="")
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        txt.insert("1.0", summary_text(sol))
        txt.configure(state="disabled")
        sel_tree.delete(*sel_tree.get_children())
        rows = [("Solid (existing)", sol["solid"])] + list(sol["alternatives"].items())
        for name, rr in rows:
            sel_tree.insert("", "end", values=[name] + [c[2].format(rr[c[0]]) for c in TABLE_COLS],
                            tags=("opt",) if name.startswith("OPTIMUM") else ())
        sel_tree.tag_configure("opt", background="#fde9b5")
        fill_tree()
        show_plots()
        o = sol["optimum"]
        status.set(f"Done. {sol['n_feasible']:,} feasible of {sol['n_grid']:,}. "
                   + (f"Max savings {o['cost_savings']:.2%} at Do {o['Do']:.4f} / Di {o['Di']:.4f} in."
                      if o else "No feasible design."))

    def reset():
        for key, _, _ in INPUT_FIELDS:
            vars_[key].set(fmt_val(getattr(defaults, key)))
        ray_var.set(defaults.require_rayleigh)
        run()

    def export():
        if not state["sol"]:
            messagebox.showinfo("Export", "Run the optimizer first.")
            return
        folder = filedialog.askdirectory(title="Choose a folder for the results")
        if not folder:
            return
        write_outputs(state["sol"], folder, state["figs"])
        messagebox.showinfo("Export", f"Saved summary, CSVs and plots to:\n{folder}")

    def on_close():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, run)
    if os.environ.get("SHAFT_GUI_TAB"):          # testing hook: open a given tab
        root.after(1500, lambda: nb.select(int(os.environ["SHAFT_GUI_TAB"])))
    root.mainloop()


# ==================================================================================
def app_dir():
    """Folder next to the .exe when frozen by PyInstaller, else next to this script."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def main():
    if "--nogui" in sys.argv:
        if sys.stdout is None:  # windowed .exe has no console
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        else:  # Windows consoles default to cp1252; keep Unicode output from crashing
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        import matplotlib
        matplotlib.use("Agg")
        sol = solve(Inputs())
        print(summary_text(sol))
        out = os.path.join(app_dir(), "results")
        write_outputs(sol, out, make_figures(sol) if sol["optimum"] else None)
        print(f"\nFiles written to {out}")
    else:
        import matplotlib
        matplotlib.use("TkAgg")
        run_gui()


if __name__ == "__main__":
    main()
