# Chui's Conjecture on the Mean Field of Unit Charges on the Circle: Statement, Attempts, and State of the Art

## 1. The conjecture and its physical content

Let $\mathbb D$ be the unit disc, $\mathbb T = \partial\mathbb D$, and $dm_2 = \pi^{-1}dx\,dy$ the normalized planar measure. In 1971 C. K. Chui, in a two-page note in the *American Mathematical Monthly* [1], asked a question that can be stated either as a problem in electrostatics or as a problem about rational functions — the two formulations are the same. The conjecture was formulated as an electrostatic minimization problem, but its real origin lies in the theory of polynomial approximation with constrained zeros; we return to that origin in Section 2.

**Physical formulation.** Place $N$ unit point charges at points $a_1,\dots,a_N \in \mathbb T$, interacting in the plane (two-dimensional Coulomb law: force inversely proportional to distance, logarithmic potential $U(z) = -\sum_k \log|z-a_k|$). The complex force exerted on a unit test charge at $z \in \mathbb D$ is

$$F(z) = \sum_{k=1}^N \frac{a_k - z}{|a_k - z|^2} = \overline{\sum_{k=1}^N \frac{1}{z-a_k}},$$

so the field strength is $|F(z)| = \big|\sum_k (z-a_k)^{-1}\big|$. Chui asked: **how should the $N$ charges be arranged on $\mathbb T$ so as to minimize the mean field strength**

$$\mathcal E(a_1,\dots,a_N) := \int_{\mathbb D} \Big|\sum_{k=1}^N \frac{1}{z-a_k}\Big|\, dm_2(z)?$$

His conjecture is that the minimum is attained precisely when the charges are **equispaced** on $\mathbb T$.

**Analytic formulation.** A rational function of the form

$$r(z) = \sum_{k=1}^N \frac{1}{z-a_k}$$

is called a **simplest fraction** (also: simple partial fraction). It is the logarithmic derivative of $P(z) = \prod_{k=1}^N (z-a_k)$, and simultaneously the Cauchy transform of the discrete measure $\sum_k \delta_{a_k}$. Writing

$$\Psi_N(z) := \sum_{k=0}^{N-1} \frac{1}{z-e^{2\pi i k/N}} = \frac{N z^{N-1}}{z^N - 1},$$

**Chui's conjecture (1971)** asserts that for every $N \ge 1$ and every $a_1,\dots,a_N \in \mathbb T$,

$$\Big\|\sum_{k=1}^N \frac{1}{z-a_k}\Big\|_{L^1(\mathbb D, dm_2)} \;\ge\; \|\Psi_N\|_{L^1(\mathbb D, dm_2)}.$$

The case $N=1$ is vacuous (any single point is "equispaced"); the first nontrivial case is $N=2$.

For the record, the only $L^1$-norm known in closed form is the one-pole value, which is elementary: with $a=1$, the region of the disc seen under the pole is $\{z: |z-1| < 1\} = \{1 + r e^{i\omega}: r < -2\cos\omega\}$, so

$$\Big\|\frac{1}{z-1}\Big\|_{L^1(\mathbb D)} = \frac{1}{\pi}\int_{\pi/2}^{3\pi/2}\int_0^{-2\cos\omega} \frac{1}{r}\, r\,dr\,d\omega = \frac{1}{\pi}\int_{\pi/2}^{3\pi/2}(-2\cos\omega)\,d\omega = \frac{4}{\pi}.$$

For $N \ge 2$ no closed form for $\|\Psi_N\|_{L^1}$ is known, and the exact value of $\lim_{N\to\infty}\|\Psi_N\|_{L^1}$ appears to be open as well. What *is* known is that $\|\Psi_N\|_{L^1} \ge C > 0$ with an absolute constant $C$ [11, (1.1)] — a fact with deep consequences, as we shall see.

It is worth placing the "equispaced" guess in its tradition. Chui's conjecture belongs to a broad family of results on $\mathbb T$ in which the roots of unity are the extremal configuration: the roots of unity maximize the Vandermonde product (equivalently, the Fekete points of the circle), they minimize the logarithmic energy $\sum_{j\ne k}\log\frac{1}{|e^{i\theta_j}-e^{i\theta_k}|}$, and equidistribution is the extremal answer to a host of Fejér–Turán-type questions on power sums of roots on the circle. "Equispaced charges" is thus not an accidental guess but an instance of a recurring theme of potential theory — with the specific twist, in Chui's problem, that the relevant quantity is not a pairwise potential energy but a *global* $L^1$ norm of a Cauchy transform.

## 2. The 1971 context: why Chui was asking

The conjecture was not an isolated curiosity. It arose inside the problem of **polynomial approximation with restricted zeros**, initiated by G. R. Mac Lane in 1949 [3]: for a bounded simply connected Jordan domain $\Omega$ with rectifiable boundary, $\partial\Omega$ is a *polynomial approximation set* relative to $\Omega$ — every zero-free holomorphic function on $\Omega$ is uniformly approximable on compacta by polynomials whose zeros all lie on $\partial\Omega$. M. Thompson [4], Chui [6], and Z. Rubinstein and E. B. Saff [7] strengthened this to *bounded* approximation in the disc.

The right global formulation is due to J. Korevaar [5], who in 1964 (in the language of potential theory: "asymptotically neutral distributions of electrons") proved the following. For a bounded simply connected domain $G$ and a set $E \subset \mathbb C$ with $E \cap G = \varnothing$, the following are equivalent:

1. $E$ is a polynomial approximation set relative to $G$;
2. $z \mapsto (z-w)^{-1}$ can be approximated locally uniformly in $G$ by polynomials with zeros only on $E$, for each $w \in E$;
3. there exist finite families $\{a_{N,k}\} \subset E$ such that $\sum_{k<N} (z-a_{N,k})^{-1} \to 0$ locally uniformly in $G$ as $N \to \infty$;
4. $\operatorname{clos} E$ separates the plane and $G$ lies in a bounded connected component of $\mathbb C \setminus \operatorname{clos}E$.

In other words: **polynomial approximation with prescribed zeros is equivalent to approximating the zero function by simplest fractions with poles on $E$**. Taking $E = \mathbb T$, $G = \mathbb D$, the question becomes: is the set

$$\mathcal{SF} := \Big\{\sum_{k=1}^N \frac{1}{z-a_k} : N \ge 1,\ a_k \in \mathbb T\Big\}$$

dense in the Bergman space $A^1(\mathbb D)$ of holomorphic $L^1$ functions? Chui suspected *no*, and his conjecture is the quantitative mechanism: since $\|\Psi_N\|_{L^1} \ge C > 0$ uniformly in $N$, the conjecture would imply that $\mathcal{SF}$ stays at positive distance from $0$ along the equispaced sequence, and in fact that $\mathcal{SF}$ is not dense in $A^1$.

Chui's subsequent work mapped the density question in weighted $L^1$-type spaces [8]: in the **Bers spaces** — weighted $L^1$ spaces with weight $\lambda_{\mathbb D}^{2-q}$, where $\lambda_{\mathbb D}$ is the Poincaré metric — the estimate $\|\Psi_N\|_{L^1} \ge C$ already implies non-density in $\mathbb D$ for $1 < q \le 2$, while Chui proved density for $q > 2$ in every Jordan domain. The borderline $q = 2$ (plain $L^1$) is exactly where his conjecture lives.

The narrative arc of the next sections is a progression from *existence* to *extremality*: Mac Lane and Korevaar asked whether simplest fractions can approximate analytic functions at all; Chui asked how small a simplest fraction can actually be; Newman answered that they cannot be arbitrarily small; ABF determined, in Hilbert spaces, that the extremal configuration is equispaced; and the original $L^1$ extremal problem remains open. This is why the conjecture survived the 1972 solution of the density question: Newman's theorem settled the first question, while Chui's conjecture is a statement about the *shape* of the extremizer, which no lower bound can address.

## 3. Newman's 1972 answer — and what it does not answer

The conjecture attracted immediate attention. In 1972 D. J. Newman published "A lower bound for an area integral" [2], proving that for *every* configuration $a_1,\dots,a_N \in \mathbb T$,

$$\int_{\mathbb D} \Big|\sum_{k=1}^N \frac{1}{z-a_k}\Big|\, dm_2(z) \;\ge\; \frac{\pi}{18}.$$

**Newman's bound.** This is a *configuration-free* lower bound: the mean field strength is never small, no matter how the charges are arranged. It settles Chui's underlying density question ($\mathcal{SF}$ is not dense in $A^1$) in the affirmative — which was, historically, the conjecture's original objective — but it is strictly weaker than the conjecture itself: it identifies no extremal configuration, and its constant $\pi/18 \approx 0.175$ is far below the equispaced values ($\|\Psi_1\|_{L^1} = 4/\pi \approx 1.273$, and $\|\Psi_N\|_{L^1}$ is known to stay bounded away from zero uniformly in $N$).

The method, as later analyses have reconstructed it, is geometric rather than functional-analytic: one projects the field onto two-dimensional planes through the origin and performs a local analysis in small discs tangent to $\mathbb T$ at each charge, exploiting the fact that the radial component of the field of a boundary charge points, on average, outward — an observation that is essentially a restatement of the positivity of the Poisson kernel, $\operatorname{Re}\frac{w+z}{w-z} \ge 0$ for $|w|=1$, $|z|<1$. The 2026 work of Doubtsov–Tselishchev–Vasilyev [17] is, in its own words, "computations similar to what is done in Newman [1972]" carried to higher dimensions and arbitrary charges; this is the only published account of the mechanism.

## 4. Why the original conjecture resists

Five decades of work make the structural obstacle clear, and it is worth stating for the functional analyst. One can almost divide the methods brought to bear on the problem into three eras: a *geometric* era (Newman's local positivity, tangent-disc arguments), an *analytic/one-parameter* era (Nazarov's solution of $N=2$), and a *structural* era (ABF, where Hilbert-space geometry converts the problem into a convexity question). This classification is not merely chronological: it explains why progress occurred in the form it did, and why the original conjecture remains difficult. The structural obstacle is as follows.

**$L^2$ is a convexity problem; $L^1$ is not.** In a Hilbert space of holomorphic functions the squared norm of a sum expands into diagonal plus *pairwise* terms:

$$\Big\|\sum_{k=1}^N \frac{1}{z-e^{i\vartheta_k}}\Big\|^2 = N\Big\|\frac{1}{z-1}\Big\|^2 + \sum_{j \ne k} \operatorname{Re}\Big\langle \frac{1}{z-e^{i\vartheta_j}}, \frac{1}{z-e^{i\vartheta_k}}\Big\rangle,$$

and, as Abakumov–Borichev–Fedorovskiy computed [11], each cross term depends only on the angular difference $\vartheta_j - \vartheta_k$ through a single kernel (in weighted Bergman spaces $A^2_{(g)}$ with weight $g$):

$$\operatorname{Re}\Big\langle \frac{1}{z-e^{i\vartheta_1}}, \frac{1}{z-e^{i\vartheta_2}}\Big\rangle_{A^2_{(g)}} = \kappa_g\, \varphi_{(g)}(\vartheta_2 - \vartheta_1),$$
$$ \qquad \varphi_{(g)}(t) = \sum_{k\ge 0} c_{(g),k}\cos((k+1)t),$$
$$\quad c_{(g),k} = \int_0^1 s^k g(1-s)\,ds.$$

The whole problem then collapses to two lemmas: (i) $\varphi_{(g)}$ is **strictly convex** on $(0,2\pi)$ (proved via the Poisson-type integral representation $\varphi_{(g)}(t) = \int_0^1 \frac{\cos t - s}{1+s^2 - 2s\cos t}\, g(1-s)\,ds$, and for the borderline weight $g(t)=t$ via Bari's Fejér-kernel identity for decreasing convex sequences [20]); (ii) a general convex-analysis fact — for a $2\pi$-periodic, even, strictly convex $\varphi$, the sum $\sum_{j\ne k}\varphi(\vartheta_j - \vartheta_k)$ is minimized, uniquely up to rotation, by the equispaced configuration (a Jensen argument applied to each orbit of the step-$s$ map on the circle). This is the complete solution of the $L^2$ version; see §5.

In $L^1$, by contrast, there is no such expansion: $|\sum f_k|$ knows about *global* cancellations between all $N$ terms, and no pairwise kernel controls it. Newman's bound is the best one can do by local (tangent-disc) analysis, and it is configuration-blind. The gap between the two is precisely the gap between a lower bound and an extremal statement.

The $N=2$ case is a useful diagnostic of the difficulty. It is a one-parameter problem: $\phi(\theta) = \int_{\mathbb D}\big|\frac{1}{z-1} + \frac{1}{z-e^{i\theta}}\big|\,dm_2(z)$, and the conjecture says $\phi$ decreases on $(0,\pi)$ and increases on $(\pi, 2\pi)$. Naive differentiation under the integral sign fails — the differentiated integrand is singular at the poles. In 2023 F. Nazarov [16] proved the $N=2$ case with an argument that is, by his own account, "a bit involved": one first moves the poles and the variable by a Möbius transformation $\zeta \mapsto (\zeta-\varepsilon)/(1-\varepsilon\zeta)$ so that all integrals stay absolutely convergent and differentiation in $\varepsilon$ at $\varepsilon = 0$ is legitimate; the resulting integral is then attacked by (i) symmetry/antisymmetry cancellations with respect to the vertical line through the two poles, (ii) two applications of Cauchy–Schwarz, (iii) a geometric lemma about a point moving along a horizontal line relative to a vertical segment (proved with the median-length/parallelogram identity), and (iv) a final polar-coordinate computation centered at a pole, which evaluates exactly to zero. The upshot: **the two-charge case of an "elementary" conjecture required a full evening of graduate-level machinery** — a strong hint that general $N$ in $L^1$ is a different kind of problem.

## 5. The Hilbert space solution: Abakumov–Borichev–Fedorovskiy

The decisive advance came in 2020–2021, when Abakumov, Borichev, and Fedorovskiy [11] (Math. Ann. 379 (2021), 1507–1532; arXiv:2009.01898) proved Chui's conjecture in the **Hilbert space setting**, i.e. with the $L^1$ norm replaced by the norm of a weighted Bergman space. Recall that $A^2_{(g)}$ consists of $f \in \operatorname{Hol}(\mathbb D)$ with $\|f\|_{(g)}^2 = \kappa_g \int_{\mathbb D} |f|^2 g(1-|z|^2)\,dm_2 < \infty$, and that the single fraction $(z-\lambda)^{-1}$, $\lambda \in \mathbb T$, belongs to $A^2_{(g)}$ iff $\int_0^1 g(s)\,ds/s < \infty$.

**Theorem (ABF, Theorem 1).** *Let $g \not\equiv 0$ be concave, non-decreasing on $[0,1]$, with $g(0)=0$ and $\int_0^1 g(s)\,ds/s < \infty$. Then for every $N \ge 1$ and every $a_1,\dots,a_N \in \mathbb T$,*

$$\Big\|\sum_{k=1}^N \frac{1}{z-a_k}\Big\|_{(g)} \;\ge\; \|\Psi_N\|_{(g)},$$

*with equality if and only if the points $a_k$ are equispaced on $\mathbb T$.*

In particular (Corollary 2) the conjecture holds in $A^2_\alpha$ for every $0 < \alpha \le 1$. The proof is exactly the convexity mechanism of §4: the Gram expansion reduces the problem to the strict convexity of $\varphi_{(g)}$ (Lemma 12, which also shows $\varphi_\alpha$ is strictly convex on $(0,2\pi)$ **if and only if** $0 < \alpha \le 1$) and the equispacing minimization lemma (Lemma 13). The "only if" in the convexity lemma is the real content: for $\alpha > 1$ the kernel fails to be convex, and indeed the equispaced configuration is no longer known to be optimal in $A^2_\alpha$ for $\alpha > 1$ — though ABF show it is optimal *asymptotically up to a constant* (Theorem 6: for $\alpha > 1$, $\min \|\sum (z-a_k)^{-1}\|_\alpha^2 \asymp N^{1-\alpha}$).

The same paper contains results that matter beyond the conjecture:

- **Sharp asymptotics (Theorem 3).** For every $\alpha > 0$,
$$\lim_{N\to\infty} N^{\alpha-1}\|\Psi_N\|_\alpha^2 = \Gamma(\alpha+2)\,\zeta(\alpha+1).$$
The proof is a model of exact asymptotic computation: the angular integral $\int_0^{2\pi} dt/|1-xe^{it}|^2 = 2\pi/(1-x^2)$ (Poisson kernel), the substitution $r = e^{-s/(2N)}$, dominated convergence, and the geometric-series evaluation of $\int_0^\infty s^\alpha/(e^s-1)\,ds$. In particular, in $A^2_1$, $\lim_{N\to\infty}\|\Psi_N\|_1 = \pi/\sqrt3$ (this $\|\cdot\|_1$ is the *Bergman* norm, not the $L^1$ norm of the conjecture), and the sequence $N^{\alpha-1}\|\Psi_N\|_\alpha^2$ is monotonically increasing.

- **The closure dichotomy (Theorem 7).** For admissible $g$, $\operatorname{clos}_{A^2_{(g)}}\mathcal{SF}$ is either $\mathcal{SF}$ itself (if $t = O(g(t))$ as $t \to 0$) or the whole space (if $g(t) = o(t)$): the simplest fractions are closed and nowhere dense in $A^2_\alpha$ for $0 < \alpha \le 1$, and **dense** in $A^2_\alpha$ for $\alpha > 1$. An "all-or-nothing" phenomenon — from the numerical analyst's viewpoint, a clean stability/instability boundary for this family of rational approximants.

- **The net theorem (Theorem 8).** For every $f \in A^2_1$, $\lim_{N\to\infty} \operatorname{dist}_{A^2_1}(f, \mathcal{SF}_N) = \pi/\sqrt3$, where $\mathcal{SF}_N$ is the set of simplest fractions with exactly $N$ poles. So $\mathcal{SF}$ is a $(\pi/\sqrt3 + \varepsilon)$-net in $A^2_1$ for every $\varepsilon > 0$, but not a $(\pi/\sqrt3 - \varepsilon)$-net: the covering radius of the family is known exactly, and the extremal elements are (up to sign) the $\Psi_N$ themselves.

- **An $L^p$ version of Thompson's theorem (Theorem 10).** For every $f \in H^\infty$, $\varepsilon, \beta > 0$, and compact $K \Subset \mathbb D$, there is $h \in \mathcal{SF}_N$ with $\|f-h\|_{L^\infty(K)} \le \varepsilon$ and, for all $0 < r < 1$,
$$\int_0^1 |h(re^{2\pi i s})|^p\,ds \le (1+\beta)\int_0^1 |\Psi_N(re^{2\pi i s})|^p\,ds + \rho(\beta) C_0^p \|f\|_{H^\infty}^p \log^p\frac{e}{1-r},$$
an $L^p$-quantitative improvement of Thompson's 1967 dominated approximation result [4]: the constructed approximant has average growth along circles no worse (up to a logarithm) than the model simplest fraction $\Psi_N$.

## 6. Beyond unit charges and beyond the plane: Doubtsov–Tselishchev–Vasilyev (2026)

The most recent attempt on the problem from the "physical" side is arXiv:2603.05233 [17] (Doubtsov, Tselishchev, Vasilyev, 2026), "Weighted Chui's conjecture". Their questions: what if the charges are *unequal* positive masses $\alpha_k$, and what if they sit on the unit sphere $\mathbb S^{d-1} \subset \mathbb R^d$ rather than the circle? (The $d=3$ case is the physically natural one, but as the authors note, formulating an analogue of Chui's conjecture there is itself difficult — one does not know what the optimal distribution on the sphere should look like; the problem is entangled with the Fekete-point and Thomson-problem questions of potential theory.) What they prove instead is a **configuration-free** bound, i.e. a multidimensional Newman bound.

**Theorem 1.1 [17].** *For $d \ge 2$, $x_1,\dots,x_n \in \mathbb S^{d-1}$, $\alpha_1,\dots,\alpha_n > 0$,*

$$\int_{\mathbb B^d} \Big|\sum_{k=1}^n \alpha_k \frac{x_k - x}{|x_k - x|^d}\Big|\, dm_2(x) \;\ge\; c_d\, \frac{\sum_{k=1}^n \alpha_k^{1+2/d}}{\sum_{k=1}^n \alpha_k^{2/d}},$$

*with $c_d > 0$ depending only on $d$. For $d=2$ and unit charges this is exactly Newman's bound.* The proof generalizes Newman's tangent-disc mechanism: for each charge one takes a ball $Q_k$ of radius $r_k = 2^{-(d+2)}\alpha_k^{2/d}/(\sum_j \alpha_j^{2/d})$ tangent to $\mathbb S^{d-1}$ at $x_k$; the key estimate is the $d$-dimensional Poisson-kernel inequality $\big\langle \frac{y-x}{|y-x|^d}, x\big\rangle \ge -\frac{1}{2|y-x|^{d-2}}$ (Lemma 2.1, equivalent in 2D to $\operatorname{Re}\frac{w+z}{w-z} \ge 0$), sharpened inside $Q_k$ (Lemma 2.2), plus a geometric comparison of distances to two tangent balls (Lemma 2.3). The integral splits into a positive main term and a remainder $B \ge 0$ proved by selecting, at each point $x$, the *dominant* tangent ball $Q_k$.

Three companion results sharpen the picture:

- **Sharpness in $d=2$ (Theorem 1.3).** For every $n$ and every positive $\alpha_1,\dots,\alpha_n$ there *exist* points $z_1,\dots,z_n \in \mathbb T$ such that $\int_{\mathbb D}\big|\sum_k \alpha_k/(z-z_k)\big|\,dm_2 \le C\, \frac{\sum \alpha_k^2}{\sum \alpha_k}$ — i.e. the lower bound of Theorem 1.1 is of optimal order in two dimensions, attained by configurations chosen to maximize cancellations.
- **A weighted Cauchy-transform bound (Theorem 1.2).** For $\nu = \sum_k \alpha_k \delta_{z_k}$, $\|\mathcal C\nu\|_{L^1(\mathbb D)} \ge C\, \frac{\sum \alpha_k^2}{\|\nu\|}$.
- **Positivity is essential (Proposition 1.4).** For $a,b \in \overline{\mathbb D}$ with $\delta = |a-b|$,
$$\int_{\mathbb D} \Big|\frac{1}{z-a} - \frac{1}{z-b}\Big|\, dm_2(z) \lesssim \delta + \delta\log\frac{1}{\delta} \to 0.$$
Two *oppositely signed* charges can cancel almost completely; the Newman-type bound is a statement about like-signed charges, and the physical reading is that like charges "cannot compensate each other" on average.

The paper also settles a related question with a PDE argument (Lemma 4.1): if the poles $z_k$ are allowed inside $\mathbb D$, then by distributional integration by parts ($\Delta u = 2\pi\sum_k \delta_{z_k}$ for $u = \sum_k \log|z-z_k|$, tested against a $1$-Lipschitz function supported in $\mathbb D$),

$$2\pi \sum_{k=1}^n \operatorname{dist}(z_k, \mathbb T) \;\le\; \int_{\mathbb D} \Big|\sum_{k=1}^n \frac{1}{z-z_k}\Big|\, dm_2(z)$$

— a clean lower bound in terms of how far the poles are from the boundary. The open problems listed include: the Newman bound with poles inside the disc (known under smallness or uniform-boundary-distance assumptions, by pushing poles to $\mathbb T$ and controlling the error via Proposition 1.4), and the higher-dimensional extremal question, where the $d=3$ unit-charge bound was an open problem in Arribas's 2024 thesis [18].

## 7. Numerical evidence

For $N \ge 3$ the conjecture becomes a genuinely multi-dimensional optimization problem over the configuration space of the angles, and it is natural to ask what brute force says. We report a direct numerical investigation, with the caveats that belong to it.

**Setup.** The integrand $|\sum_k (z-a_k)^{-1}|$ is singular at the boundary poles, so absolute values carry a quadrature error of order $10^{-2}$–$10^{-3}$; only *comparative* claims are robust. Accordingly, every configuration below — optimizer candidates, grid points, and the equispaced reference alike — is evaluated with the **same** quadrature (Gauss–Legendre in $r$, midpoint rule in $\theta$), so that the shared error cancels in the comparison. Following the standard trick, the complex disc grid $z = r e^{i\theta}$ and the area weights are precomputed once, and only the poles move inside the objective; by rotation invariance one pole is fixed at angle $0$. Two complementary searches are used: (i) the gradient-free Nelder–Mead method with multiple random restarts, and (ii) an *exhaustive* grid over the configuration space (40×40 grid at 9° resolution for $N=3$, 15×15×15 at 24° for $N=4$), which rules out local minima (Python 3.13 venv, numpy 2.5 / scipy 1.18).

**Results.**

| $N$ | optimizer's minimum (angles, degrees) | excess over equispaced (same quadrature) |
|---|---|---|
| 3 | $0^\circ,\ 119.9^\circ,\ 239.4^\circ$ (8 restarts) | $-0.15\%$ |
| 4 | $0^\circ,\ 90.00^\circ,\ 180.00^\circ,\ 269.7^\circ$ (6 restarts) | $+0.005\%$ |

| $N$ | exhaustive grid minimum (angles, degrees) | excess over equispaced (same quadrature) |
|---|---|---|
| 3 | $0^\circ,\ 126^\circ,\ 243^\circ$ (40×40, 9°) | $+0.27\%$ |
| 4 | $0^\circ,\ 96^\circ,\ 192^\circ,\ 264^\circ$ (15³, 24°) | $+3.31\%$ |

In both cases the optimizer converges, up to rotation, to the equispaced configuration, and its best value agrees with the equispaced value within the quadrature noise (the same quadrature reproduces independently computed equispaced values to $0.07$–$0.28\%$; the negative "excess" at $N=3$ is of exactly this size and is not a counterexample). The exhaustive grids agree with a stronger statement: in both cases the grid minimum is precisely the on-grid configuration *closest* to the equispaced one — for $N=3$ the ideal gaps of $13\tfrac{1}{3}$ grid steps cannot be realized on a 40-point grid, and the observed gaps $14,13,13$ (and its mirror image $0^\circ,117^\circ,234^\circ$, which attains the identical value) are the closest possible; for $N=4$ the exact equispaced configuration is off-grid ($90^\circ$ is not a multiple of $24^\circ$), and the observed minimum is again the nearest on-grid configuration, its excess being consistent with the rise away from the minimum plus the coarser quadrature. Deliberately clumped adversarial configurations are decisively worse at $N=3$: e.g. $(0^\circ,9^\circ,180^\circ)$ gives $2.51$ and $(0^\circ,9^\circ,18^\circ)$ gives $3.61$, against $1.81$ for the equispaced configuration — an excess of $39\%$ to $99\%$. No configuration found by either search beats the equispaced one.

The equispaced $L^1$ norms themselves, computed with a higher-resolution quadrature (the angular integral reduces to the AGM formula below), grow monotonically:

$$\|\Psi_1\|_{L^1} = \frac{4}{\pi} = 1.27324 \ \text{(exact)}, \quad \|\Psi_2\| = 1.63255, \quad \|\Psi_3\| = 1.80934, \quad \|\Psi_4\| = 1.91474,$$

$$\|\Psi_{10}\| = 2.14325, \quad \|\Psi_{100}\| = 2.31188, \quad \|\Psi_{1000}\| = 2.33037, \quad \|\Psi_{5000}\| = 2.33203,$$

where the angular integral was evaluated via

$$\int_0^{2\pi}\frac{d\theta}{|1-xe^{i\theta}|} = \frac{2\pi}{(1+x)\,\mathrm{AGM}\!\left(1,\frac{1-x}{1+x}\right)},$$

The sequence appears to converge, at a rate consistent with the $\|\Psi_N\|_{L^1} \ge C$ lower bound, to a limit near $2.332$ — roughly thirteen times Newman's universal bound $\pi/18 \approx 0.175$, and unrelated to the $A^2_1$-limit $\pi/\sqrt3 \approx 1.814$, which belongs to a different norm.

The numerical picture — equispaced as the minimizer found, at every $N$ tested, both by unconstrained multi-start local search and by exhaustive grid search — is consistent with the $N=2$ theorem [16] and the ABF result [11], and with the conjecture. It is evidence, not a proof: the grids have finite resolution, the search explores a finite set of restarts of a local method, and the $L^1$ landscape could in principle harbor a clumped competitor that neither has yet found.

## 8. Other threads

The conjecture sits in a small ecosystem of results on simplest fractions:

- **Borodin [12]** (2016) and **Borodin–Shklyaev [13]** (2023): approximation by simple partial fractions with constrained poles; density of quantized approximations.
- **Komarov [14]** (2023): a Newman-type bound for $L^p[-1,1]$-means of the logarithmic derivative of polynomials with all zeros on the unit circle — the same object, a different averaging domain (the diameter instead of the disc), closer to the territory of numerical analysis.
- **Chui–Zhong [15]** (2023): order of uniform approximation by polynomial interpolation, with the electrostatic field estimates feeding the analysis.
- **Anderson–Eiderman [10]** (2006): solved the Macintyre–Fuchs problem on the growth of the Hausdorff content of level sets of simplest fractions — evidence that the objects are rich beyond approximation theory.
- **Chui–Shen [9]** (1985): order of approximation by electrostatic fields, the quantitative companion to Chui's 1973 density results.

One more development deserves a cautious footnote. In August 2026 it was reported in the press [21], citing the support network of the imprisoned Russian mathematician Azat Miftakhov, that a paper "La conjecture de Chui dans un cadre hilbertien" appeared in the *Revue de la filière mathématique*, claiming Chui's conjecture "in a Hilbert space". Since the Bergman-space ($L^2$) version was already settled by [11] in 2021, any such claim would have to concern a different Hilbert-space setting; at the time of writing the result is not independently verified, and the original $L^1$ conjecture — the 1971 question as stated — should still be regarded as **open for $N \ge 3$**.

## 9. State of the art and open problems

Summarizing what is known about the original conjecture (the $L^1$ statement, general $N$). One subtle point deserves to be stated explicitly, because it explains the conjecture's survival: Chui's *original* objective and the *modern* objective are not the same. In 1971 the conjecture was a route toward understanding the density of $\mathcal{SF}$ in $A^1$; today, after Newman, the density issue is no longer the obstacle, and the conjecture survives because it asks for the **extremal configuration** — not merely a positive lower bound.

| Regime | Status |
|---|---|
| $N = 1$ | Trivial; $\|\Psi_1\|_{L^1} = 4/\pi$ exactly |
| $N = 2$ | **Proved** (equispaced = antipodal minimizes) — Nazarov [16] |
| $N \ge 3$, $L^1(\mathbb D)$ | **Open** |
| $L^2$ version, $A^2_\alpha$, $0 < \alpha \le 1$ | **Proved** with uniqueness — ABF [11] |
| $L^2$ version, $A^2_\alpha$, $\alpha > 1$ | Open; asymptotically optimal up to a constant [11, Thm 6] |
| Configuration-free lower bounds | $\pi/18$ in $L^1$ (Newman [2]); sharp-order weighted bounds in $\mathbb R^d$ [17] |
| Numerical optimization, $L^1$ | $N=3,4$: multi-start Nelder–Mead and exhaustive grids both minimize at the (on-grid) equispaced configuration, within quadrature noise (§7) |

Open problems worth naming:

1. **The original conjecture for $N \ge 3$ in $L^1$.** The convexity machinery of [11] breaks exactly at the $L^1$ level; the $N=2$ proof of [16] is a one-parameter argument with no visible route to $N$. The numerical evidence of §7 (multi-start optimization and exhaustive grids at $N=3,4$) supports the conjecture but cannot replace a proof.
2. **The $L^2_\alpha$ case $\alpha > 1$**: the kernel $\varphi_\alpha$ loses convexity; is equispacing still optimal, or does a "clumped" configuration win?
3. **Poles inside the disc** (Newman bound and conjecture with $a_k \in \mathbb D$): partially open [17], with Lemma 4.1 giving the boundary-distance term.
4. **Higher dimensions**: formulate the conjecture on $\mathbb S^{d-1}$ (optimal configurations — Fekete/Thomson-type) and prove the weighted bounds for charges in $\mathbb B^d$.
5. **The exact $L^1$ norm of $\Psi_N$** and the limit $\lim_{N\to\infty}\|\Psi_N\|_{L^1}$: unknown in closed form; Newman's bound and the $A^2_1$-limit $\pi/\sqrt3$ bracket the landscape from different norms.

The arc of the attempts is itself the lesson: Newman's 1972 tangent-disc argument is a *local* method that sees no configuration; Nazarov's 2023 proof is a *one-parameter* method that sees the configuration but not the mechanism; ABF's 2021 proof is a *structural* method — the Hilbert-space norm turns the question into a strictly convex pairwise interaction, and equispacing falls out of Jensen's inequality. The original conjecture is what remains when the structure is stripped away: a global $L^1$ cancellation problem for the Cauchy transform of discrete boundary measures, fifty-five years after Chui's two-page note.

## References

1. C. K. Chui, *A lower bound of fields due to unit point masses*, Amer. Math. Monthly **78** (1971), no. 7, 779–780.
2. D. J. Newman, *A lower bound for an area integral*, Amer. Math. Monthly **79** (1972), no. 9, 1015–1016.
3. G. R. Mac Lane, *Polynomials with zeros on a rectifiable Jordan curve*, Duke Math. J. **16** (1949), 461–477.
4. M. Thompson, *Approximation of bounded analytic functions on the disc*, Nieuw. Arch. Wisk. (3) **15** (1967), 49–54.
5. J. Korevaar, *Asymptotically neutral distributions of electrons and polynomial approximation*, Ann. of Math. **80** (1964), no. 3, 403–410.
6. C. K. Chui, *Bounded approximation by polynomials whose zeros lie on a circle*, Trans. Amer. Math. Soc. **138** (1969), 171–182.
7. Z. Rubinstein, E. B. Saff, *Bounded approximation by polynomials whose zeros lie on a circle*, Proc. Amer. Math. Soc. **29** (1971), 482–486.
8. C. K. Chui, *On approximation in the Bers spaces*, Proc. Amer. Math. Soc. **40** (1973), no. 2, 438–442.
9. C. K. Chui, X. C. Shen, *Order of approximation by electrostatic fields due to electrons*, Constr. Approx. **1** (1985), no. 2, 121–135.
10. J. M. Anderson, V. Ya. Eiderman, *Cauchy transforms of point masses: the logarithmic derivative of polynomials*, Ann. of Math. **163** (2006), 1057–1076.
11. E. Abakumov, A. Borichev, K. Fedorovskiy, *Chui's conjecture in Bergman spaces*, Math. Ann. **379** (2021), 1507–1532; arXiv:2009.01898.
12. P. A. Borodin, *Approximation by simple partial fractions with constraints on the poles. II*, Sb. Math. **207** (2016), no. 3–4, 331–341.
13. P. A. Borodin, K. S. Shklyaev, *Density of quantized approximations*, Russ. Math. Surv. **78** (2023), no. 5, 797–851.
14. M. A. Komarov, *A Newman type bound for $L_p[-1,1]$-means of the logarithmic derivative of polynomials having all zeros on the unit circle*, Constr. Approx. **58** (2023), no. 3, 551–563.
15. C. K. Chui, L. Zhong, *Order of uniform approximation by polynomial interpolation in the complex plane and beyond*, Indag. Math. (N.S.) **34** (2023), no. 2, 418–456.
16. F. Nazarov, answer to "How to prove that $\phi'(\theta) < 0$ for $\theta \in (0,\pi)$?", MathOverflow, question 451462 (2023).
17. E. Doubtsov, A. Tselishchev, I. Vasilyev, *Weighted Chui's conjecture*, arXiv:2603.05233 (2026).
18. D. Arribas, *Minimal energy on the circle*, MSc thesis, Universitat de Barcelona (2024).
19. H. Hedenmalm, B. Korenblum, K. Zhu, *Theory of Bergman Spaces*, GTM 199, Springer, 2000.
20. N. K. Bary, *A Treatise on Trigonometric Series*, Vol. I, Pergamon, 1964 (Ch. 1, §30).
21. Meduza, "Jailed Russian mathematician Azat Miftakhov publishes paper solving Chui's Conjecture in French journal" (Aug. 21, 2026) — press report, not independently verified.

---
