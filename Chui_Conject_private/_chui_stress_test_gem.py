"""
Corrected Iterated Dynamics for the Two-Pole Polarization Lemma.
Uses local 1D energy minimization instead of boundary collisions.
"""
import numpy as np
from scipy.optimize import minimize_scalar

# --- Static grid ---
R, T = 150, 1024
nodes, weights = np.polynomial.legendre.leggauss(R)
rr = nodes * 0.5 + 0.5
ww = weights * 0.5
theta = (np.arange(T) + 0.5) * 2 * np.pi / T
z = rr[:, None] * np.exp(1j * theta)[None, :]
area_w = (ww * rr)[:, None] * (2 * np.pi / T) / np.pi 

def L1_pair_eval(d, m, bg_poles):
    """Evaluates L1 norm for a background + an active pair at m +/- d."""
    active = [m - d, m + d]
    f = np.zeros((R, T), dtype=complex)
    for a in np.concatenate([bg_poles, active]):
        f += 1.0 / (z - np.exp(1j * a))
    return float(np.sum(np.abs(f) * area_w))

rng = np.random.default_rng(42)

print("=== Iterated Polarization Dynamics (Local Minimum Search) ===")

def iterated_dynamics_local(N, max_steps=50):
    angles = np.sort(rng.uniform(0, 2*np.pi, N))
    
    for step in range(max_steps):
        gaps = np.diff(angles)
        gaps = np.append(gaps, 2*np.pi - angles[-1] + angles[0])
        
        # Check convergence to equispaced
        if np.max(gaps) - np.min(gaps) < 1e-3:
            return step, True, angles
            
        # Target the pair with the smallest gap
        i = np.argmin(gaps)
        j = (i + 1) % N
        m = (angles[i] + gaps[i]/2.0) % (2*np.pi)
        
        # Identify background poles
        bg_mask = np.ones(N, dtype=bool)
        bg_mask[i] = False
        bg_mask[j] = False
        bg_poles = angles[bg_mask]
        
        # Determine strict collision bounds with nearest neighbors
        left_neighbor = angles[(i - 1) % N]
        right_neighbor = angles[(j + 1) % N]
        dist_left = (m - left_neighbor) % (2*np.pi)
        dist_right = (right_neighbor - m) % (2*np.pi)
        d_max = min(dist_left, dist_right) / 2.0
        
        # Find the exact L1 minimum within the free gap (avoiding collision)
        res = minimize_scalar(
            L1_pair_eval, 
            bounds=(0.01, d_max - 0.01), 
            args=(m, bg_poles), 
            method='bounded'
        )
        
        # Update angles to the optimal separation
        angles[i] = (m - res.x) % (2*np.pi)
        angles[j] = (m + res.x) % (2*np.pi)
        angles = np.sort(angles)
        
    return max_steps, False, angles

for N in [3, 4, 5]:
    successes = 0
    trials = 20
    avg_steps = 0
    
    for _ in range(trials):
        steps, converged, final_angles = iterated_dynamics_local(N)
        if converged:
            successes += 1
            avg_steps += steps
            
    if successes > 0:
        avg_steps /= successes
    print(f"N={N} | Trials: {trials} | Converged to Equispaced: {successes}/{trials} | Avg Steps: {avg_steps:.1f}")