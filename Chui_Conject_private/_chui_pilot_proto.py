"""PROTOTYPE of the 'rearrangement era' pilot proposed in Chui_essay_v1_private_r3_Cla.md.

Idea under test (the review's central, checkable sub-claim):
  Polarizing a pair of poles toward antipodal (increase their angular separation,
  midpoint fixed) decreases the super-level set area  A(t) = m2({z in D: |f(z)| > t})
  for every t simultaneously (pointwise), or at least decreases its integral
  (= the L1 norm).

Geometry note (checked, corrects the review's sketch):
  The poles a_k lie on the BOUNDARY T = dD, so f = sum 1/(z-a_k) is HOLOMORPHIC on
  the open disc D. Hence |f| is a genuine subharmonic function on all of D (finite),
  and log|f| is subharmonic too. The super-level sets {|f|>t} for large t are small
  boundary-anchored caps (one per pole), which grow inward and merge as t decreases.
  So the relevant topology is boundary caps, not interior islands.

This is a FEASIBILITY PROTOTYPE, not the full pilot: small grid, few configs, N=3.
It answers: (a) is the pilot affordable? (b) first read on the monotonicity claim.
"""
import numpy as np

# --- static grid (precomputed once) ---
R, T = 150, 1024
nodes, weights = np.polynomial.legendre.leggauss(R)
rr = nodes * 0.5 + 0.5
ww = weights * 0.5
theta = (np.arange(T) + 0.5) * 2 * np.pi / T
z = rr[:, None] * np.exp(1j * theta)[None, :]                       # (R, T)
area_w = ((ww * rr)[:, None] * (2 * np.pi / T)[None, :]) / np.pi    # (R, T), per-cell area

def field_abs(angles):
    f = np.zeros((R, T), dtype=complex)
    for a in angles:
        f += 1.0 / (z - np.exp(1j * a))
    return np.abs(f)

def L1(F):
    return float(np.sum(F * area_w))

def A_at(t, F):
    """A(t) = m2({|F| > t}) directly: sum of cell areas where |F| > t."""
    return float(np.sum(area_w[F > t]))

def polarize_pair(angles, i, j, dprime):
    """Replace pair (i,j) by (m - d', m + d'), midpoint m fixed."""
    out = list(angles)
    m = (angles[i] + angles[j]) / 2.0
    out[i] = (m - dprime) % (2 * np.pi)
    out[j] = (m + dprime) % (2 * np.pi)
    return out

rng = np.random.default_rng(1234)
NCFG = 10
print(f"Prototype: N=3, R={R}, T={T}, {NCFG} random configs, pole 1 fixed at 0.\n", flush=True)

# common t-grid for pointwise comparison: quantiles of a representative |f|
sample = field_abs([0.0, 1.0, 4.0])
t_common = np.quantile(sample, np.linspace(0.02, 0.98, 25))

n_pointwise_ok = 0
n_l1_ok = 0
n_steps = 0
viol_max = 0.0
viol_examples = []
for c in range(NCFG):
    th2 = rng.uniform(0, 2 * np.pi)
    th3 = rng.uniform(0, 2 * np.pi)
    base = [0.0, th2, th3]
    F0 = field_abs(base)
    L0 = L1(F0)
    m = (th2 + th3) / 2.0
    d = abs(th2 - th3) / 2.0
    d = min(d, np.pi - d)                      # half-separation in (0, pi]
    d = min(d, np.pi / 2)                      # cap: separation <= pi
    for dprime in [min(d + 0.3, np.pi / 2), np.pi / 2]:   # small step, full step
        if dprime <= d + 1e-12:
            continue
        pol = polarize_pair(base, 1, 2, dprime)
        F1 = field_abs(pol)
        L1v = L1(F1)
        n_steps += 1
        A0 = np.array([A_at(t, F0) for t in t_common])
        A1 = np.array([A_at(t, F1) for t in t_common])
        viol = float(np.max(A1 - A0))
        viol_max = max(viol_max, viol)
        if viol > 1e-9:
            viol_examples.append((c, dprime, viol))
        n_pointwise_ok += (viol <= 1e-9)
        n_l1_ok += (L1v <= L0 + 1e-9)

print(f"polarization steps evaluated: {n_steps}")
print(f"pointwise A(t) decreased for ALL t:  {n_pointwise_ok}/{n_steps}")
print(f"L1 norm decreased:                    {n_l1_ok}/{n_steps}")
print(f"max pointwise violation (A_after - A_before): {viol_max:.3e}")
print(f"  (positive = area INCREASED somewhere -> counterexample to pointwise monotonicity)")
if viol_examples:
    print("  examples (config, d', violation):", viol_examples[:5])
print("\nPrototype done. See plan for the full pilot specification.", flush=True)
