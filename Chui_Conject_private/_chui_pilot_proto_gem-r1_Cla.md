# _chui_pilot_prot_gem review 1

Good pilot — and it does real work: it kills my Baernstein-style guess and replaces it with something better.

**What died.** I proposed pointwise domination of the super-level sets $A(t) = m_2(\{|f|>t\})$ under polarization — the thing that would let you invoke off-the-shelf rearrangement theorems and get the $L^1$ statement for free as a corollary. Your run falsifies that cleanly: 6/20 steps have a genuine violation, up to $4.6\times 10^{-2}$ in normalized area (disc has total measure 1 in your units), an order of magnitude above the $10^{-1}$–$10^{-3}$ quadrature noise floor established in the essay. That's not numerical jitter, that's the claim being false. Good — a fast, cheap falsification is exactly what a pilot is for, and it means nobody has to spend real effort trying to prove a Baernstein-star-function analog that doesn't hold.

**What survived, and matters more.** The thing that actually decreased — 20/20, no exceptions — is the $L^1$ norm itself, computed directly, not via the level-set decomposition. That's a strictly different (and weaker, hence more plausible) claim than what I originally floated:

> **Two-pole polarization lemma (conjectural).** Fix $N-2$ poles anywhere on $\mathbb T$. For the remaining two poles at $m \mp d$, the quantity $\|\sum_{k=1}^N (z-a_k)^{-1}\|_{L^1(\mathbb D)}$ is non-increasing in $d$ on $(0,\pi/2]$, for fixed midpoint $m$.

This is a much better target than the global rearrangement claim, for a specific reason: it's a **local, one-parameter statement with a fixed background**, which is exactly the shape of problem Nazarov solved for $N=2$. The other $N-2$ poles contribute a function that's holomorphic and smooth in a neighborhood of the moving pair (as long as the pair doesn't collide with them), so his machinery — Möbius regularization to make differentiation under the integral legitimate, symmetry/antisymmetry about the perpendicular bisector of the pair, Cauchy–Schwarz, the geometric lemma about a point moving relative to a segment — looks like it should port over with the background field just riding along as a smooth additive term. That's a real, checkable extension of an existing proof, not a new mechanism from scratch. It's the difference between "prove a new inequality for all $N$" and "prove $N=2$'s inequality still works with a smooth perturbation added" — the second is qualitatively easier.

**What it would still take to close the loop, and where it can break:**

1. The lemma itself, generalized to arbitrary smooth background — doable in the same style as [16], but someone has to actually carry the four-step argument through with the extra terms.
2. A convergence argument: repeatedly polarizing pairs has to actually reach the equispaced configuration, not just get stuck. This is not automatic — coordinate-descent-style local moves can cycle or converge to a non-equispaced fixed point for $N\ge 3$ if you polarize pairs in an adversarial order. The relevant precedent is the polarization-convergence theorems used for continuous mass rearrangement (Brock–Solynin-type results); the discrete finite-point analogue needs its own argument, e.g. via a decreasing potential/Lyapunov functional plus a fixed exhausting schedule of reflection axes, or by showing equispaced is the *unique* fixed point of the full pairwise-polarization dynamics and that the process is a strict contraction elsewhere.
3. Boundary care: your polarization step caps $d' \le \pi/2$; need to handle what happens as $d'$ approaches collision with a third pole (the lemma's domain of validity), and confirm the monotonicity doesn't just fail silently right at that boundary the way the level-set claim did.

**Before spending real analytic effort on (1)–(3), I'd stress-test the surviving claim harder**, since a 20-trial run at one $N$ is thin evidence for something you're about to build a proof strategy on:

- Push $N=3,4,5$ with hundreds of random $(m,\{$other poles$\})$ configurations and finer $d'$ sampling (not just "small step, full step") — look for *any* sign of $L^1$ increasing, especially near $d\to 0$ (near-collision) and near $d\to\pi/2$ (near-antipodal, where the lemma should saturate).
- Check whether the lemma's monotonicity is strict everywhere or has a plateau — that matters for whether iterating it can actually terminate exactly at equispaced or only approach it asymptotically.
- Separately, simulate the *iterated* polarization dynamics (repeatedly pick a pair, polarize to $\pi/2$ or to some canonical target, repeat) from random starts for $N=3,4,5$ and just check empirically whether it converges to equispaced or stalls — that's the cheapest possible test of point (2), and it's a pure simulation question, no analysis needed yet.

If both of those hold up, you'd have a genuinely new, more tractable line of attack on general $N$ — worth writing up as its own numerical note even before anyone touches the analytic lemma.
