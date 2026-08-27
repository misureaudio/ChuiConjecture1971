import sys
import numpy as np

def AGM(a, b):
    """Vectorized AGM."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # FIX 1: Increased tolerance to 1e-14 to prevent infinite ULP-oscillation loops
    while np.max(np.abs(a - b)) > 1e-14 * np.max(np.maximum(a, b)):
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
    """
    FIX 3: Applied Change of Variables x = r^N. 
    Integral becomes (1/pi) int_0^1 x^{1/N} I_ang(x) dx.
    This avoids underflow and peak undersampling for large N.
    """
    x = (np.arange(M) + 0.5) / M
    return float(np.trapezoid(x**(1/N) * I_ang_agm(x), x) / np.pi)

print(flush=True)
print("N       ||Psi_N||_L1(D)  (equispaced poles -- Chui's candidate)", flush=True)
for N in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
    print(f"{N:<7} {psi_N_L1(N):.8f}", flush=True)

print(flush=True)
print("4/pi      =", 4 / np.pi, " (exact, N=1: polar coords about the pole)", flush=True)
print("pi/sqrt(3)=", np.pi / np.sqrt(3), " (ABF Thm 3, alpha=1: ||Psi_N||^2 -> pi^2/3)", flush=True)
print("pi/18     =", np.pi / 18, " (Newman's universal lower bound)", flush=True)

# ---------- two-charge profile: poles at e^{+-i phi} ----------
def two_charge_L1(phi, R=300, T=1 << 12):
    nodes, weights = np.polynomial.legendre.leggauss(R)
    rr = nodes * 0.5 + 0.5
    ww = weights * 0.5
    t = (np.arange(T) + 0.5) * 2 * np.pi / T
    
    # FIX 2: Evaluate using complex arrays. This completely avoids algebraic 
    # errors that were causing spurious internal singularities and division by zero.
    z = rr[:, None] * np.exp(1j * t)[None, :]
    f_val = 1.0 / (z - np.exp(-1j * phi)) + 1.0 / (z - np.exp(1j * phi))
    
    inner = np.trapezoid(np.abs(f_val), t, axis=1)
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