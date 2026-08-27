"""Coarse-grid numerical check of Chui's conjecture for N=3 (and coarser N=4).

Design notes (for reliability, given the boundary singularity |z-a|^{-1}):
- All configurations are evaluated with the SAME quadrature (Gauss-Legendre in r,
  midpoint in theta), so systematic quadrature error is shared and COMPARATIVE
  statements (grid min vs equispaced) are much more robust than absolute values.
- Rotation invariance: pole 1 fixed at angle 0.
- Validation: N=2 and N=3 equispaced values must match the AGM-based run of
  _chui_check_gem.py (1.63254888 / 1.80934360) to ~1e-3.
- (v2, per chui_check_v1_r2_gem.md: explicit superposition of the N charges per
  configuration; np.atleast_2d so single configs and grid matrices work alike.)
"""
import numpy as np
import time

def config_L1(angles, R, T, chunk=16):
    """L1 norm (normalized measure) of sum_k 1/(z - e^{i ang_k}) over D."""
    # Force 2D so single configurations and grid matrices are handled uniformly
    angles = np.atleast_2d(angles)
    P, N_charges = angles.shape

    nodes, weights = np.polynomial.legendre.leggauss(R)
    rr = nodes * 0.5 + 0.5
    ww = weights * 0.5
    t = (np.arange(T) + 0.5) * 2 * np.pi / T
    z = rr[:, None] * np.exp(1j * t)[None, :]          # (R, T)

    out = np.empty(P)
    for s in range(0, P, chunk):
        ang = angles[s:s + chunk]
        chunk_size = len(ang)
        f = np.zeros((chunk_size, R, T), dtype=complex)

        # Loop over each configuration in the chunk
        for k in range(chunk_size):
            # Evaluate the N_charges for this specific configuration
            poles = np.exp(1j * ang[k])
            for p in poles:
                f[k] += 1.0 / (z - p)  # Superposition of fields!

        inner = np.trapezoid(np.abs(f), t, axis=2)     # (chunk, R)
        out[s:s + chunk] = np.sum(inner * (ww * rr)[None, :], axis=1) / np.pi

    return out

# ---------------- validation ----------------
Rv, Tv = 300, 4096
print("Running validations...", flush=True)
v2 = config_L1(np.array([0.0, np.pi]), Rv, Tv)[0]
v3 = config_L1(np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3]), Rv, Tv)[0]
print(f"validation N=2 equispaced: {v2:.8f}  (gem run: 1.63254888)", flush=True)
print(f"validation N=3 equispaced: {v3:.8f}  (gem run: 1.80934360)", flush=True)

# ---------------- N = 3 grid ----------------
M, R, T = 40, 160, 3072
g = np.linspace(0, 2 * np.pi, M, endpoint=False)
a2, a3 = np.meshgrid(g, g, indexing='ij')
angs = np.column_stack([np.zeros(M * M), a2.ravel(), a3.ravel()])

print(flush=True)
print(f"N=3 grid: {M}x{M} = {M*M} configurations, R={R}, T={T}", flush=True)
t0 = time.time()
vals = config_L1(angs, R, T)
t1 = time.time()

i_min = int(np.argmin(vals))
equisp = config_L1(np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3]), R, T)[0]

print(f"Computed in {t1-t0:.2f} seconds.", flush=True)
print(f"grid min = {vals[i_min]:.8f} at angles(deg) {np.degrees(angs[i_min])}", flush=True)
print(f"equispaced (same quadrature) = {equisp:.8f}", flush=True)
print(f"excess of grid min over equispaced: {vals[i_min]-equisp:.3e}"
      f" ({(vals[i_min]-equisp)/equisp*100:.4f}%)", flush=True)

print(flush=True)
adv = {
    "clump (0, 9deg, 180deg)":  [0.0, np.pi / 20, np.pi],
    "clump (0, 9deg, 189deg)":  [0.0, np.pi / 20, np.pi + np.pi / 20],
    "two close (0, 9deg, 120deg)": [0.0, np.pi / 20, 2 * np.pi / 3],
    "triple clump (0, 9deg, 18deg)": [0.0, np.pi / 20, np.pi / 10],
    "near-equisp (0, 117deg, 234deg)": [0.0, 117 * np.pi / 180, 234 * np.pi / 180],
}
for name, a in adv.items():
    print(f"adversarial {name}: {config_L1(np.array(a), R, T)[0]:.8f}", flush=True)

# ---------------- N = 4 grid (coarser) ----------------
M4, R4, T4 = 15, 100, 1024
g4 = np.linspace(0, 2 * np.pi, M4, endpoint=False)
b2, b3, b4 = np.meshgrid(g4, g4, g4, indexing='ij')
angs4 = np.column_stack([np.zeros(M4 ** 3), b2.ravel(), b3.ravel(), b4.ravel()])

print(flush=True)
print(f"N=4 grid: {M4}^3 = {M4**3} configurations, R={R4}, T={T4}", flush=True)
t0 = time.time()
vals4 = config_L1(angs4, R4, T4)
t1 = time.time()

i_min4 = int(np.argmin(vals4))
equisp4 = config_L1(np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]), R4, T4)[0]

print(f"Computed in {t1-t0:.2f} seconds.", flush=True)
print(f"grid min = {vals4[i_min4]:.8f} at angles(deg) {np.degrees(angs4[i_min4])}", flush=True)
print(f"equispaced (same quadrature) = {equisp4:.8f}", flush=True)
print(f"excess of grid min over equispaced: {vals4[i_min4]-equisp4:.3e}"
      f" ({(vals4[i_min4]-equisp4)/equisp4*100:.4f}%)", flush=True)
print("DONE", flush=True)
