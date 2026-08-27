"""N=3 (and N=4) optimization check for Chui's conjecture.

Strategy (per chui_check_v1_r1_gem.md): precompute the complex disc grid and
area weights ONCE; only the poles move inside the objective. Rotation invariance
fixes pole 1 at angle 0. Nelder-Mead with multiple random restarts.

Reliability notes:
- The quadrature (Gauss-Legendre in r, midpoint in theta) is the SAME for every
  configuration, so comparisons (optimizer min vs equispaced) are robust.
- Validation: N=2 and N=3 equispaced values must match the AGM-based run of
  _chui_check_gem.py (1.63254888 / 1.80934360) to ~1e-3.
"""
import numpy as np
from scipy.optimize import minimize

# 1. Precompute the integration grid ONCE
R, T = 300, 1024
nodes, weights = np.polynomial.legendre.leggauss(R)
rr = nodes * 0.5 + 0.5
ww = weights * 0.5
t = (np.arange(T) + 0.5) * 2 * np.pi / T
z_grid = rr[:, None] * np.exp(1j * t)[None, :]
area_weights = (ww[:, None] * rr[:, None]) * (2 * np.pi / T) / np.pi

def N_charge_L1(phis):
    f_val = np.zeros_like(z_grid, dtype=complex)
    for phi in phis:
        f_val += 1.0 / (z_grid - np.exp(1j * phi))
    return float(np.sum(np.abs(f_val) * area_weights))

# ---------------- validation ----------------
v2 = N_charge_L1([0.0, np.pi])
v3 = N_charge_L1([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
v4 = N_charge_L1([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
print(f"validation N=2 equispaced: {v2:.8f}  (gem run: 1.63254888)", flush=True)
print(f"validation N=3 equispaced: {v3:.8f}  (gem run: 1.80934360)", flush=True)
print(f"validation N=4 equispaced: {v4:.8f}", flush=True)

def optimize_N(N, n_restarts=8, seed=0):
    rng = np.random.default_rng(seed)
    best_x, best_f = None, np.inf
    for r in range(n_restarts):
        x0 = np.sort(rng.uniform(0, 2 * np.pi, N - 1))
        res = minimize(lambda x: N_charge_L1([0.0, *x]), x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-12})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x
    return best_f, best_x

# ---------------- N = 3 ----------------
f3, x3 = optimize_N(3)
angs3 = [0.0, *x3]
print(flush=True)
print(f"N=3 optimizer: min = {f3:.8f}", flush=True)
print(f"  angles (deg): {np.degrees(angs3)}", flush=True)
print(f"  equispaced (same quadrature): {v3:.8f}", flush=True)
print(f"  excess over equispaced: {f3 - v3:.3e} ({(f3 - v3) / v3 * 100:.4f}%)", flush=True)

# ---------------- N = 4 ----------------
f4, x4 = optimize_N(4, n_restarts=6)
angs4 = [0.0, *x4]
print(flush=True)
print(f"N=4 optimizer: min = {f4:.8f}", flush=True)
print(f"  angles (deg): {np.degrees(angs4)}", flush=True)
print(f"  equispaced (same quadrature): {v4:.8f}", flush=True)
print(f"  excess over equispaced: {f4 - v4:.3e} ({(f4 - v4) / v4 * 100:.4f}%)", flush=True)
print("DONE", flush=True)
