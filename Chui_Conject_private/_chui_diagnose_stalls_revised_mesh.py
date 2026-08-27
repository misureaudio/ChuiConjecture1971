"""
Mesh-Refinement Study for Chui Conjecture Dynamics.
Testing the aliasing / quadrature noise hypothesis raised in review.
"""
import numpy as np

def L1_eval_arr(d, mid, b_poles, z, area_w):
    # helper for the fast-forward
    f = np.zeros(z.shape, dtype=complex)
    for a in np.concatenate([b_poles, [mid - d, mid + d]]):
        f += 1.0 / (z - np.exp(1j * a))
    return float(np.sum(np.abs(f) * area_w))

# Reproducible random state to ensure we scan the exact same topography
rng = np.random.default_rng(101)
N_test = 4
init_angles = np.sort(rng.uniform(0, 2*np.pi, N_test))

# Pick a pair to polarize
i, j = 0, 1
m = (init_angles[i] + (init_angles[j] - init_angles[i] + 2*np.pi) % (2*np.pi) / 2.0) % (2*np.pi)
bg_poles = init_angles[2:]
dist_left = (m - init_angles[-1]) % (2*np.pi)
dist_right = (init_angles[2] - m) % (2*np.pi)
d_max = min(dist_left, dist_right) / 2.0

print("=== PART 1: 1D Topography Scan (Aliasing vs Reality) ===")
N_SAMPLES = 40
d_samples = np.linspace(0.005, d_max - 0.005, N_SAMPLES)
d_spacing = d_samples[1] - d_samples[0]
print(f"Sampling d at {N_SAMPLES} points.")
print(f"d-sample spacing: {d_spacing:.5f} rad")
print("-" * 70)

# We will store the final stalled state from T=1024 to use in Part 2
stalled_angles = None 

for T in [1024, 2048, 4096, 8192]:
    grid_spacing = 2 * np.pi / T
    
    R = 150
    nodes, weights = np.polynomial.legendre.leggauss(R)
    rr = nodes * 0.5 + 0.5
    ww = weights * 0.5
    theta = (np.arange(T) + 0.5) * 2 * np.pi / T
    z = rr[:, None] * np.exp(1j * theta)[None, :]
    area_w = (ww * rr)[:, None] * (2 * np.pi / T) / np.pi

    def eval_pair(d):
        return L1_eval_arr(d, m, bg_poles, z, area_w)

    vals = np.array([eval_pair(d) for d in d_samples])
    min_idx = np.argmin(vals)
    diffs = np.diff(vals)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    
    # Calculate depth relative to the boundary (d_max)
    depth = vals[-1] - vals[min_idx]
    
    print(f"T={T:<4d} (grid:{grid_spacing:.5f}) | Sign changes: {sign_changes:<2d} | "
          f"Min at d={d_samples[min_idx]:.4f} | Depth below d_max: {depth:+.2e}")
    
    # Fast-forward a greedy descent at T=1024 to get a reproducible trapped state
    if T == 1024:
        stalled_angles = init_angles.copy()
        for _ in range(15):  # 15 steps is enough to hit the trap
            gaps = np.diff(stalled_angles)
            gaps = np.append(gaps, 2*np.pi - stalled_angles[-1] + stalled_angles[0])
            idx = np.argmin(gaps)
            nxt = (idx + 1) % N_test
            mid = (stalled_angles[idx] + gaps[idx]/2.0) % (2*np.pi)
            
            bg = np.ones(N_test, dtype=bool)
            bg[idx] = False; bg[nxt] = False
            b_poles = stalled_angles[bg]
            
            dl = (mid - stalled_angles[(idx - 1) % N_test]) % (2*np.pi)
            dr = (stalled_angles[(nxt + 1) % N_test] - mid) % (2*np.pi)
            d_mx = min(dl, dr) / 2.0
            
            g_ds = np.linspace(0.005, d_mx - 0.005, 40)
            g_vs = [L1_eval_arr(g_ds_i, mid, b_poles, z, area_w) for g_ds_i in g_ds]
            best_d = g_ds[np.argmin(g_vs)]
            
            stalled_angles[idx] = (mid - best_d) % (2*np.pi)
            stalled_angles[nxt] = (mid + best_d) % (2*np.pi)
            stalled_angles = np.sort(stalled_angles)

print("\n=== PART 2: The 'Ridge' Perturbation Test (Signal vs Reality) ===")
eps = 0.02
print(f"Using stalled configuration from T=1024: {np.degrees(stalled_angles).astype(int)} deg")
print(f"Perturbing pole 1 by eps={eps} ({np.degrees(eps):.1f} deg) — safely wider than the grid.")
print("-" * 70)

for T in [1024, 2048, 4096, 8192]:
    R = 150
    nodes, weights = np.polynomial.legendre.leggauss(R)
    rr = nodes * 0.5 + 0.5
    ww = weights * 0.5
    theta = (np.arange(T) + 0.5) * 2 * np.pi / T
    z = rr[:, None] * np.exp(1j * theta)[None, :]
    area_w = (ww * rr)[:, None] * (2 * np.pi / T) / np.pi
    
    def L1_eval(angles):
        f = np.zeros((R, T), dtype=complex)
        for a in angles:
            f += 1.0 / (z - np.exp(1j * a))
        return float(np.sum(np.abs(f) * area_w))

    base_L1 = L1_eval(stalled_angles)
    
    pert_angles = stalled_angles.copy()
    pert_angles[1] = (pert_angles[1] + eps) % (2*np.pi)
    pert_L1 = L1_eval(pert_angles)
    
    diff = pert_L1 - base_L1
    print(f"T={T:<4d} | Base L1: {base_L1:.6f} | Perturbed diff: {diff:+.2e}")
    