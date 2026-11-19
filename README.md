# Hollow Shaft Optimizer

MECH.4730 solid vs. hollow shaft equivalence. The app finds the hollow steel shaft (outside diameter Do, inside diameter Di) that **maximizes cost savings** while being at least as strong, stiff and vibration-safe as the existing 3.0 in solid shaft (90 in span, 2000 RPM, E = 30×10⁶ psi).

## Download and run (Windows)

1. Open the **Releases** page of this repo and download `HollowShaftOptimizer.exe` from the newest build.
2. Double-click it. No Python install is needed.

Windows SmartScreen may warn that the app is unrecognized, because the exe is not code-signed. Click **More info → Run anyway**.

Every push to `main` builds a new exe automatically (see `.github/workflows/build-exe.yml`). Pull requests also build it; the exe is under the workflow run's **Artifacts**.

## Run from source

```
py -m pip install -r requirements.txt
py shaft_optimizer.py            # GUI
py shaft_optimizer.py --nogui    # console + files in .\results
```

Or double-click `install.bat` once, then `run.bat`.

## How it works

1. **Grid sweep:** every (Do, Di) on a 0.005 in grid (about 222,000 designs) is checked against all constraints.
2. **Optimizer:** SciPy SLSQP, seeded from the best grid point, finds the exact maximum-savings design, which is then re-checked against every constraint.

| Constraint | Check |
|---|---|
| Envelope | Do ≤ tunnel − clearance (4.0 − 0.05 in) |
| Weight | A/A₀ ≤ 0.50, A = π/4 (Do² − Di²) |
| Cost | savings = 1 − 1.2·A/A₀ ≥ 30% |
| Strength | Z/Z₀ ≥ 1, Z = π(Do⁴ − Di⁴)/(32·Do) (torsion identical, J = 2I) |
| Stiffness | I/I₀ ≥ 1, self-weight sag ≤ solid |
| Critical speed | Nc ≥ 1.25 × 2000 RPM, ωn = (π/L)²√(EIg/ρA) (Rayleigh optional) |
| Wall | Do/t ≤ 20 |

All inputs and limits are editable in the window. Results can be exported as CSV, a text summary and PNG plots.

## Default result

| Design | Do / Di (in) | Savings | Strength vs. solid | Nc (RPM) |
|---|---|---|---|---|
| Solid (existing) | 3.000 / 0 | – | 1.00 | 1775 |
| Max savings | 3.950 / 3.420 | 47.9% | 1.000 | 3091 |
| Max margin at 50% weight | 3.950 / 3.332 | 40.0% | 1.127 | 3057 |
