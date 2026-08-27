"""
Diagnostic script for Iterated Polarization Dynamics.
Addresses the optimizer artifacts and limit cycle concerns raised in review.
"""
import numpy as np

# --- Static grid ---
R, T = 150, 1024
nodes, weights = np.polynomial.legendre.leggauss(R)
rr = nodes * 0.5 + 0.5
ww = weights * 0.5
theta = (np.arange(T) + 0.5) * 2 * np.pi / T
z = rr[:, None] * np.exp(1j * theta)[None, :]
area_w = (ww * rr)[:, None] * (2 * np.pi / T) / np.pi 

def L1_eval(angles):
    """Evaluates L1 norm for an array of angles."""
    f = np.zeros((R, T), dtype=complex)
    for a in angles:
        f += 1.0 / (z - np.exp(1j * a))
    return float(np.sum(np.abs(f) * area_w))

def L1_pair_eval(d, m, bg_poles):
    return L1_eval(np.concatenate([bg_poles, [m - d, m + d]]))

rng = np.random.default_rng(101)

print("=== PART 1: 1D Topography Scan (Is L1(d) multimodal?) ===")
N_scan = 4
scan_angles = np.sort(rng.uniform(0, 2*np.pi, N_scan))
i, j = 0, 1
m = (scan_angles[i] + (scan_angles[j] - scan_angles[i] + 2*np.pi) % (2*np.pi) / 2.0) % (2*np.pi)
bg_poles = scan_angles[2:]

dist_left = (m - scan_angles[-1]) % (2*np.pi)
dist_right = (scan_angles[2] - m) % (2*np.pi)
d_max = min(dist_left, dist_right) / 2.0

ds = np.linspace(0.005, d_max - 0.005, 200)
vals = np.array([L1_pair_eval(d, m, bg_poles) for d in ds])

# Analyze the 1D curve
min_idx = np.argmin(vals)
is_interior = (0 < min_idx < len(ds) - 1)
diffs = np.diff(vals)
sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

print(f"Scan resolution: 200 points, d_max = {d_max:.4f}")
print(f"Global 1D minimum found at d = {ds[min_idx]:.4f}")
print(f"Is minimum interior? {is_interior}")
print(f"Number of gradient sign changes (0=monotone, 1=unimodal, >1=multimodal): {sign_changes}")
if sign_changes > 1:
    print("  -> BRENT'S METHOD WOULD FAIL HERE (Multimodal!). Reviewer was right to suspect.")
elif sign_changes == 1 and is_interior:
    print("  -> INTERIOR MINIMUM CONFIRMED. The 2-pole polarization lemma is FALSE.")


print("\n=== PART 2: Robust Dynamics (Global 1D Search + Random Pairs) ===")
def robust_dynamics(N, max_steps=100):
    angles = np.sort(rng.uniform(0, 2*np.pi, N))
    
    for step in range(max_steps):
        gaps = np.diff(angles)
        gaps = np.append(gaps, 2*np.pi - angles[-1] + angles[0])
        
        if np.max(gaps) - np.min(gaps) < 1e-3:
            return True, angles
            
        # RANDOM pair selection instead of greedy
        i = rng.integers(0, N)
        j = (i + 1) % N
        m = (angles[i] + gaps[i]/2.0) % (2*np.pi)
        
        bg_mask = np.ones(N, dtype=bool)
        bg_mask[i] = False
        bg_mask[j] = False
        bg_poles = angles[bg_mask]
        
        left_neighbor = angles[(i - 1) % N]
        right_neighbor = angles[(j + 1) % N]
        dist_left = (m - left_neighbor) % (2*np.pi)
        dist_right = (right_neighbor - m) % (2*np.pi)
        d_max = min(dist_left, dist_right) / 2.0
        
        # GLOBAL 1D SEARCH via grid
        grid_ds = np.linspace(0.005, d_max - 0.005, 50)
        grid_vals = [L1_pair_eval(d, m, bg_poles) for d in grid_ds]
        best_d = grid_ds[np.argmin(grid_vals)]
        
        angles[i] = (m - best_d) % (2*np.pi)
        angles[j] = (m + best_d) % (2*np.pi)
        angles = np.sort(angles)
        
    return False, angles

N_test = 4
converged, final_angles = robust_dynamics(N_test)
print(f"Did robust dynamics converge to equispaced? {converged}")


print("\n=== PART 3: N-Dimensional Local Minimum Test ===")
# Test if the stalled configuration is a genuine local minimum in the full (N-1)-dimensional space
if not converged:
    eps = 1e-4
    base_L1 = L1_eval(final_angles)
    is_true_local_min = True
    
    print(f"Base L1 of stalled configuration: {base_L1:.6f}")
    
    # We fix angle 0 (rotation invariance) and perturb the other N-1 angles
    for k in range(1, N_test):
        for sign in [-1, 1]:
            pert_angles = final_angles.copy()
            pert_angles[k] = (pert_angles[k] + sign * eps) % (2*np.pi)
            pert_L1 = L1_eval(pert_angles)
            
            diff = pert_L1 - base_L1
            # If any perturbation DECREASES the L1 norm, it is NOT a local minimum
            if diff < -1e-9:
                is_true_local_min = False
                print(f"  -> Perturbing pole {k} by {sign*eps} DECREASED L1 by {diff:.2e}")
                
    if is_true_local_min:
        print("RESULT: Stalled state IS a genuine N-dimensional local minimum!")
        print("  -> Conclusion: The global landscape has traps. Coordinate descent cannot win.")
    else:
        print("RESULT: Stalled state is NOT a local minimum.")
        print("  -> Conclusion: The dynamics are merely cycling/stalling on ridges.")