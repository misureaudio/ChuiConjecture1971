# Chui check 2 charges

##

You have written a very nice, mathematically advanced script to test the Chui conjecture! However, there are **three distinct issues** causing it to hang, give incorrect results, and struggle with numerical precision. 

Here is exactly what is wrong and how to fix it:

### 1. The Vectorized AGM function has an Infinite Loop

In your `AGM` function, your `while` loop condition uses a tolerance of `1e-16`:

```python
while np.max(np.abs(a - b)) > 1e-16 * np.max(np.maximum(a, b)):
```

Machine epsilon for 64-bit floats is roughly `2.22e-16`. When elements in your array converge, their final difference is often 1 ULP (Unit in the Last Place), which is exactly `~2.22e-16 * a`. Because `2.22e-16 > 1e-16`, the condition evaluates to `True` forever. Because you are taking `np.max()` over an array of 131,072 elements, it is almost guaranteed that at least *one* element gets stuck oscillating in the last bit, **causing the script to hang indefinitely.**

* **Fix:** Change the tolerance to something safely above machine epsilon, such as `1e-14`.

### 2. A Mathematical Error in the Two-Charge Denominator

In `two_charge_L1`, you attempted to expand the integrand into reals. Your numerator is correct, but your denominator is wrong:

```python
# Your code:
den = np.abs(rr[:, None] * ct[None, :] - c) 
```

The true denominator is $|z^2 - 2zc + 1|$. On the unit circle ($r=1$), this happens to equal $2|\cos t - c|$. However, **inside the disk ($r < 1$), this is no longer true.** By using that formula inside the disk, you accidentally created a non-integrable line of singularities where $r \cos t = c$. This causes division-by-zero, `inf` values, and ruins both the execution time (due to subnormal float handling) and the mathematical result.

* **Fix:** It is much safer, cleaner, and faster to just evaluate the complex function directly and let NumPy handle the magnitude:

```python
z = rr[:, None] * np.exp(1j * t)[None, :]
val = np.abs(1 / (z - np.exp(-1j * phi)) + 1 / (z - np.exp(1j * phi)))
```

### 3. Precision Loss for Large $N$ (Under-sampling the Peak)

In `psi_N_L1`, you are evaluating $N r^N I_{ang}(r^N)$ on a uniform grid of $r$.
When $N$ gets large (e.g., $N=5000$), $r^{5000}$ underflows to `0.0` almost everywhere. The function spikes violently in a tiny margin near $r=1$. A uniform grid will miss this peak entirely, leading to massive numerical inaccuracies.

* **Fix:** Use a clever substitution! Let $x = r^N$.
$dx = N r^{N-1} dr \implies N r^N dr = x^{1/N} dx$.
The integral transforms beautifully into: $\frac{1}{\pi} \int_0^1 x^{1/N} I_{ang}(x) dx$. 
This removes $N$ from the exponents entirely, meaning **no underflow**, and a uniform grid in $x$ will be highly accurate for *any* value of $N$.

---

### The Corrected, Fast Code

Here is the fully repaired version of your script. It should now run in just a few seconds and give highly accurate outputs.

```python
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
```
