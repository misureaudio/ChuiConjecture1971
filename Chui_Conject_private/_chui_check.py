import sys
import numpy as np

def AGM(a, b):
    """Vectorized AGM."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    while np.max(np.abs(a - b)) > 1e-16 * np.max(np.maximum(a, b)):
        a, b = (a + b) / 2.0, np.sqrt(a * b)
    return a

def I_ang_agm(x):
    """int_0^{2pi} dtheta / |1 - x e^{i theta}| = 2 pi / ((1+x) AGM(1, (1-x)/(1+x)))."""
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = 2 * np.pi / ((1.0 + x) * AGM(1.0, (1.0 - x) / (1.0 + x)))
    return np.where(x > 0, val, 2 * np.pi)

def I_ang_direct(x, M=1 << 16):
    th = (np.arange(M) + 0.5) * 2 * np.pi / M
    return np.trapezoid(1.0 / np.abs(1 - x * np.exp(1j * th)), th)

print("x      direct        AGM-formula", flush=True)
for x in [0.0, 0.25, 0.5, 0.9, 0.99, 0.999]:
    print(f"{x:<8} {I_ang_direct(x):.6f}   {I_ang_agm(x):.6f}", flush=True)

def psi_N_L1(N, M=1 << 17):
    """L1 norm of Psi_N = N z^{N-1}/(z^N - 1) over D, dm = pi^{-1} dxdy.
    L1 = (1/pi) int_0^1 N r^N I_ang(r^N) dr."""
    r = (np.arange(M) + 0.5) / M
    return float(np.trapezoid(N * r**N * I_ang_agm(r**N), r) / np.pi)

print(flush=True)
print("N       ||Psi_N||_L1(D)  (equispaced poles -- Chui's candidate)", flush=True)
for N in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
    print(f"{N:<7} {psi_N_L1(N):.8f}", flush=True)

print(flush=True)
print("4/pi      =", 4 / np.pi, " (exact, N=1: polar coords about the pole)", flush=True)
print("pi/sqrt(3)=", np.pi / np.sqrt(3), " (ABF Thm 3, alpha=1: ||Psi_N||^2 -> pi^2/3)", flush=True)
print("pi/18     =", np.pi / 18, " (Newman's universal lower bound)", flush=True)

# ---------- two-charge profile: poles at e^{+-i phi} ----------
# r(z) = 1/(z-e^{-i phi}) + 1/(z-e^{i phi}) = 2(z - c)/(z^2 - 2 z c + 1),  c = cos phi
# on z = r e^{it}: |num| = 2 sqrt(r^2 - 2 r c cos t + c^2),  |den| = 2 |r cos t - c|
def two_charge_L1(phi, R=300, T=1 << 12):
    c = np.cos(phi)
    nodes, weights = np.polynomial.legendre.leggauss(R)
    rr = nodes * 0.5 + 0.5
    ww = weights * 0.5
    t = (np.arange(T) + 0.5) * 2 * np.pi / T
    ct = np.cos(t)
    num = np.sqrt(rr[:, None]**2 - 2 * rr[:, None] * c * ct[None, :] + c**2)
    den = np.abs(rr[:, None] * ct[None, :] - c)
    inner = np.trapezoid(num / den, t, axis=1)
    return float(np.sum(ww * rr * inner) / np.pi)

print(flush=True)
print("angle theta between charges   L1 norm of r(z)", flush=True)
phis = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.5707963, 1.6, 1.7, 1.9]
vals = {p: two_charge_L1(p) for p in phis}
for p in phis:
    print(f"theta={2*p:8.5f}   {vals[p]:.8f}", flush=True)
print(flush=True)
print("expect min at theta = pi (equispaced): two-charge case of Chui's conjecture", flush=True)
print("(proved by F. Nazarov, MathOverflow 451462)", flush=True)
