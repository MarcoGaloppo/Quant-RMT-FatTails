"""

A minimal laboratory for random matrices and what fat tails do to them. My
first steps in directly adressing quantitative finance.

We know that the basic estimation problem underneath all of quantitative portfolio 
construction is: we have ``N`` assets, we got ``T`` observations, and sadly the ratio 
``q = N/T`` is not small. Interestingly, whilst sample covariance ``E = X X^T / T`` is
unbiased entry by entry , its *eigenvalues* are systematically distorted. However, here
Random Matrix Theory (RMT) lands a helping hand! We wish to thank RMT for it and employ it.
This module provides the three ingredients the notebooks need:

1. **Ensembles.** Wigner and Wishart matrices with exchangeable entry
   distributions: Gaussian, uniform, Rademacher, centred exponential (to test
   universality), Student-t and symmetric alpha-stable (to break it), plus
   elliptical returns (common daily volatility — fat tails "the simple market's way").
2. **The laws.** Semicircle and Marchenko–Pastur densities, empirical Stieltjes
   transforms, and the BBP formulas for spiked covariance models (outlier
   position, eigenvector overlap, detectability threshold).
3. **Estimators.** Namely RMT cleaning schemes we test against each other with a
   given underlyin truth: eigenvalue clipping (Laloux et al.), Ledoit–Wolf linear
   shrinkage, the Rotationally Invariant Estimator (Ledoit–Péché and Bun–Bouchaud–
   Potters, nonlinear shrinkage), and the oracle (the best any eigenvector-keeping
   estimator can ever do).

Everything is pure numpy. Conventions: returns matrices are ``(N, T)`` — assets
in rows, time in columns; eigenvalues are sorted *descending* unless stated.

Author: Marco Galoppo
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    # entry samplers / ensembles
    "sample_entries",
    "sample_stable",
    "wigner",
    "goe",
    "returns_gaussian",
    "returns_student",
    "returns_stable",
    "sample_cov",
    "sample_corr",
    # theoretical laws
    "semicircle_pdf",
    "mp_edges",
    "mp_pdf",
    "stieltjes",
    # spiked models / BBP
    "spiked_cov",
    "bbp_threshold",
    "spike_lambda",
    "spike_overlap2",
    "spike_ell_from_lambda",
    # eigen tools
    "eigh_desc",
    "ipr",
    "recompose",
    # estimators
    "xi_clip",
    "lw_shrinkage",
    "xi_rie",
    "xi_oracle",
    "clean_cov",
    # portfolio metrics
    "min_var_weights",
    "portfolio_variance",
    "frobenius",
    "risk_report",
    # Taleb-school instruments
    "day_scale",
    "max_to_sum_ratio",
    "kappa",
    "normal_scores",
]


# --------------------------------------------------------------------------- #
#  Entry distributions
# --------------------------------------------------------------------------- #
def sample_entries(
    dist: str,
    size,
    rng: np.random.Generator,
    nu: float = 3.0,
    alpha: float = 1.5,
) -> np.ndarray:
    """Draw iid entries from ``dist``, standardised to mean 0 and variance 1
    whenever the variance exists.

    dist ∈ {"gaussian", "uniform", "rademacher", "exponential", "student",
    "stable"}.  For "student" pass ``nu`` (variance exists for nu > 2, fourth
    moment for nu > 4).  For "stable" pass ``alpha`` in (0, 2): infinite
    variance, no standardisation possible — entries have unit *scale* instead.
    """
    if dist == "gaussian":
        return rng.standard_normal(size)
    if dist == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size)
    if dist == "rademacher":
        return rng.integers(0, 2, size).astype(float) * 2.0 - 1.0
    if dist == "exponential":                       # skewed, all moments finite
        return rng.exponential(1.0, size) - 1.0
    if dist == "student":
        if nu <= 2:
            raise ValueError("nu <= 2 has infinite variance; use dist='stable' "
                             "if that is what you want to study.")
        return rng.standard_t(nu, size) * np.sqrt((nu - 2.0) / nu)
    if dist == "stable":
        return sample_stable(alpha, size, rng)
    raise ValueError(f"unknown dist {dist!r}")


def sample_stable(alpha: float, size, rng: np.random.Generator) -> np.ndarray:
    """Symmetric alpha-stable samples, unit scale (Chambers–Mallows–Stuck).

    Tail: P(|X| > x) ~ C x^(-alpha).  alpha = 2 is Gaussian (up to scale);
    alpha < 2 has infinite variance; alpha <= 1 infinite mean.
    """
    if not 0.0 < alpha <= 2.0:
        raise ValueError("alpha must be in (0, 2]")
    u = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size)
    w = rng.exponential(1.0, size)
    if abs(alpha - 1.0) < 1e-12:
        return np.tan(u)
    x = (np.sin(alpha * u) / np.cos(u) ** (1.0 / alpha)
         * (np.cos((1.0 - alpha) * u) / w) ** ((1.0 - alpha) / alpha))
    return x


# --------------------------------------------------------------------------- #
#  Ensembles
# --------------------------------------------------------------------------- #
def wigner(
    n: int,
    rng: np.random.Generator,
    dist: str = "gaussian",
    nu: float = 3.0,
    alpha: float = 1.5,
) -> np.ndarray:
    """A symmetric random matrix with iid entries above (and on) the diagonal.

    Entries are drawn *once* and mirrored, so for dist="rademacher" the matrix
    literally contains only two values — and still produces the semicircle.
    Normalisation is H = M / sqrt(n) for finite-variance entries (semicircle on
    [-2, 2]), and H = M / n^(1/alpha) for "stable" (Lévy matrices, Cizeau–
    Bouchaud), for which the limiting spectral density is heavy-tailed.
    """
    m = sample_entries(dist, (n, n), rng, nu=nu, alpha=alpha)
    h = np.triu(m) + np.triu(m, 1).T
    if dist == "stable":
        return h / n ** (1.0 / alpha)
    return h / np.sqrt(n)


def goe(n: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian Orthogonal Ensemble, normalised to the semicircle on [-2, 2]."""
    return wigner(n, rng, dist="gaussian")


def _chol(cov: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if cov is None:
        return None
    cov = np.asarray(cov, float)
    if cov.shape != (n, n):
        raise ValueError("cov has wrong shape")
    return np.linalg.cholesky(cov)


def returns_gaussian(
    n: int,
    t: int,
    rng: np.random.Generator,
    cov: Optional[np.ndarray] = None,
) -> np.ndarray:
    """(n, t) Gaussian returns with population covariance ``cov`` (default I)."""
    x = rng.standard_normal((n, t))
    l = _chol(cov, n)
    return x if l is None else l @ x


def returns_student(
    n: int,
    t: int,
    nu: float,
    rng: np.random.Generator,
    cov: Optional[np.ndarray] = None,
    elliptical: bool = True,
    standardize: bool = True,
) -> np.ndarray:
    """(n, t) Student-t returns with population covariance ``cov`` (default I).

    elliptical=True  — ONE chi-square volatility per day shared by all assets:
        r_t = sqrt(nu / s_t) * L z_t,   s_t ~ chi2(nu).
        This is the multivariate t: cross-sectionally dependent even when cov=I
        (a big day is big for everyone).  The market's kind of fat tails.
    elliptical=False — every entry gets its own independent t draw: iid fat
        tails, the "textbook" kind, cross-sectionally independent when cov=I.

    standardize=True rescales by sqrt((nu-2)/nu) so the population covariance is
    exactly ``cov`` (needs nu > 2).
    """
    if nu <= 2 and standardize:
        raise ValueError("nu <= 2: variance infinite, cannot standardise")
    z = rng.standard_normal((n, t))
    l = _chol(cov, n)
    if l is not None:
        z = l @ z
    if elliptical:
        s = rng.chisquare(nu, t)
        x = z * np.sqrt(nu / s)[None, :]
    else:
        s = rng.chisquare(nu, (n, t))
        x = z * np.sqrt(nu / s)
    if standardize:
        x = x * np.sqrt((nu - 2.0) / nu)
    return x


def returns_stable(n: int, t: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """(n, t) iid symmetric alpha-stable 'returns', unit scale. No covariance
    exists; use with sample_corr and morbid curiosity."""
    return sample_stable(alpha, (n, t), rng)


def sample_cov(x: np.ndarray, demean: bool = False) -> np.ndarray:
    """E = X X^T / T."""
    x = np.asarray(x, float)
    if demean:
        x = x - x.mean(axis=1, keepdims=True)
    return x @ x.T / x.shape[1]


def sample_corr(x: np.ndarray, demean: bool = False) -> np.ndarray:
    """Sample correlation matrix: E rescaled to unit diagonal."""
    e = sample_cov(x, demean=demean)
    d = 1.0 / np.sqrt(np.diag(e))
    return e * np.outer(d, d)


# --------------------------------------------------------------------------- #
#  Theoretical laws
# --------------------------------------------------------------------------- #
def semicircle_pdf(x, radius: float = 2.0) -> np.ndarray:
    """Wigner semicircle density on [-radius, radius]."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    inside = np.abs(x) < radius
    out[inside] = 2.0 / (np.pi * radius ** 2) * np.sqrt(radius ** 2 - x[inside] ** 2)
    return out


def mp_edges(q: float, sigma2: float = 1.0) -> Tuple[float, float]:
    """Support edges of the Marchenko–Pastur law: sigma2 * (1 ∓ sqrt(q))^2."""
    return sigma2 * (1.0 - np.sqrt(q)) ** 2, sigma2 * (1.0 + np.sqrt(q)) ** 2


def mp_pdf(x, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko–Pastur density for E = X X^T / T with iid entries of variance
    sigma2 and q = N/T.

    For q <= 1 this integrates to 1.  For q > 1 (more assets than observations)
    the continuous part below integrates to 1/q and the remaining mass (1 - 1/q)
    sits in a delta at zero — E is rank T < N; the histogram will show the spike
    at 0 that this density deliberately does not contain.
    """
    x = np.asarray(x, float)
    lm, lp = mp_edges(q, sigma2)
    out = np.zeros_like(x)
    inside = (x > lm) & (x < lp)
    xi = x[inside]
    out[inside] = np.sqrt((lp - xi) * (xi - lm)) / (2.0 * np.pi * q * sigma2 * xi)
    return out


def stieltjes(z, eigvals: np.ndarray) -> np.ndarray:
    """Empirical Stieltjes transform s(z) = (1/N) sum_i 1 / (lambda_i - z),
    for complex z (scalar or array), Im(z) != 0."""
    z = np.atleast_1d(np.asarray(z, complex))
    s = np.mean(1.0 / (eigvals[None, :] - z[:, None]), axis=1)
    return s if s.size > 1 else s[0]


# --------------------------------------------------------------------------- #
#  Spiked covariance & the BBP transition
# --------------------------------------------------------------------------- #
def spiked_cov(
    n: int,
    spikes: Sequence[float],
    rng: np.random.Generator,
    market: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """C = I + sum_k (ell_k - 1) v_k v_k^T with orthonormal spike directions.

    ``spikes`` are the population eigenvalues ell_k (> 1 for genuine factors).
    If market=True the first direction is the uniform vector (the market mode:
    everything up together); the rest are random orthonormal completions.
    Returns (C, V) with V of shape (n, k) holding the spike directions.
    """
    spikes = np.asarray(spikes, float)
    k = len(spikes)
    g = rng.standard_normal((n, k))
    if market:
        g[:, 0] = 1.0
    v, _ = np.linalg.qr(g)
    if market and v[0, 0] < 0:                     # QR sign convention
        v[:, 0] *= -1.0
    c = np.eye(n) + (v * (spikes - 1.0)[None, :]) @ v.T
    return c, v


def bbp_threshold(q: float) -> float:
    """Detectability threshold: a spike ell produces an outlier iff
    ell > 1 + sqrt(q)."""
    return 1.0 + np.sqrt(q)


def spike_lambda(ell, q: float) -> np.ndarray:
    """Asymptotic position of the sample outlier for population spike ell:
    lambda = ell + q*ell/(ell-1) above threshold, else stuck at the MP edge."""
    ell = np.atleast_1d(np.asarray(ell, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            ell > bbp_threshold(q),
            ell + q * ell / (ell - 1.0),
            (1.0 + np.sqrt(q)) ** 2,
        )
    return out if out.size > 1 else out[0]


def spike_overlap2(ell, q: float) -> np.ndarray:
    """Asymptotic squared overlap |<u_sample, v_true>|^2 for spike ell:
    (1 - q/(ell-1)^2) / (1 + q/(ell-1)) above threshold, 0 below.
    """
    ell = np.atleast_1d(np.asarray(ell, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        num = 1.0 - q / (ell - 1.0) ** 2
        den = 1.0 + q / (ell - 1.0)
        ov = num / den
    out = np.where(ell > bbp_threshold(q), np.clip(ov, 0.0, 1.0), 0.0)
    return out if out.size > 1 else out[0]


# --------------------------------------------------------------------------- #
#  Eigen tools
# --------------------------------------------------------------------------- #
def spike_ell_from_lambda(lam, q: float) -> np.ndarray:
    """Invert spike_lambda: given an observed outlier position, the population
    spike that would have produced it.

        lambda = ell + q*ell/(ell-1)   =>   ell^2 - (1 + lambda - q) ell + lambda = 0

    Take the upper root (the lower one is the mirror solution below the BBP
    threshold).  Returns nan when the discriminant is negative, i.e., when no
    spike could have produced that lambda — which is exactly the sub-threshold
    case: the eigenvalue is bulk, not a shrunken factor.

    Feed it lambda/sigma2 (and the matching q) when working with an effective
    null, and read the answer as approximate: the formula assumes ONE spike in
    a white background, so several coexisting factors bias it low.
    """
    lam = np.atleast_1d(np.asarray(lam, float))
    b = 1.0 + lam - q
    disc = b ** 2 - 4.0 * lam
    with np.errstate(invalid="ignore"):
        out = np.where(disc > 0, (b + np.sqrt(np.maximum(disc, 0.0))) / 2.0, np.nan)
    return out if out.size > 1 else out[0]


def eigh_desc(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and matching eigenvectors of a symmetric m."""
    lam, u = np.linalg.eigh(m)
    return lam[::-1], u[:, ::-1]


def ipr(u: np.ndarray) -> np.ndarray:
    """Inverse participation ratio sum_i v_i^4 per column.

    Delocalised (random-direction) vectors give ~ 3/N (Porter–Thomas);
    a vector concentrated on one component gives ~ 1.  The localisation
    diagnostic for 'is this eigenvector a factor or a single fat day?'.
    """
    u = np.atleast_2d(u.T).T                       # promote 1-D to a column
    return np.sum(u ** 4, axis=0)


def recompose(u: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Rebuild U diag(xi) U^T — every estimator here keeps sample eigenvectors
    and surgically replaces eigenvalues."""
    return (u * xi[None, :]) @ u.T


# --------------------------------------------------------------------------- #
#  Estimators (the shoot-out contestants)
# --------------------------------------------------------------------------- #
def xi_clip(lam: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Eigenvalue clipping (Laloux et al. 1999).

    Keep eigenvalues above the MP edge (they might be signal), replace everything
    below by their common average.  Descending input.
    """
    lam = np.asarray(lam, float)
    _, lp = mp_edges(q, sigma2)
    xi = lam.copy()
    bulk = lam <= lp
    if bulk.any():
        xi[bulk] = lam[bulk].mean()
    return xi


def lw_shrinkage(x: np.ndarray, demean: bool = False) -> Tuple[np.ndarray, float, float]:
    """Ledoit–Wolf (2004) linear shrinkage toward mu*I, intensity from data.

    Returns (Sigma_lw, rho, mu):  Sigma = rho*mu*I + (1-rho)*E.
    Same eigenvectors as E; eigenvalues linearly squeezed toward mu — the
    optimal *linear* eigenvalue map, one straight line for all eigenvalues.
    """
    x = np.asarray(x, float)
    n, t = x.shape
    if demean:
        x = x - x.mean(axis=1, keepdims=True)
    e = x @ x.T / t
    mu = np.trace(e) / n
    d2 = np.sum((e - mu * np.eye(n)) ** 2) / n           # ||.||^2 = tr(.^2)/n
    # (1/T^2) sum_k ||x_k x_k^T - E||_F^2, vectorised over days
    s2 = np.sum(x * x, axis=0)                            # ||x_k||^2 per day
    cross = np.sum(x * (e @ x), axis=0)                   # x_k^T E x_k per day
    b2 = (np.sum(s2 ** 2) - 2.0 * np.sum(cross) + t * np.sum(e ** 2)) / (n * t ** 2)
    b2 = min(b2, d2)
    rho = b2 / d2 if d2 > 0 else 1.0
    sigma = rho * mu * np.eye(n) + (1.0 - rho) * e
    return sigma, rho, mu


def xi_rie(
    lam: np.ndarray,
    q: float,
    eta: Optional[float] = None,
    preserve_trace: bool = True,
) -> np.ndarray:
    """Rotationally Invariant Estimator (Ledoit–Péché 2011; Bun–Bouchaud–
    Potters 2017): nonlinear shrinkage computed from the observed spectrum only.

        xi_i = lambda_i / | 1 - q + q * z_i * g(z_i) |^2 ,   z_i = lambda_i - i*eta

    with g(z) = (1/N) Tr (z - E)^{-1} the Stieltjes transform of the sample
    spectrum, evaluated just below the real axis (eta ~ N^(-1/2), the resolution
    at which N eigenvalues can be told apart).  Self-term excluded to reduce
    finite-N self-interaction bias (leave-one-out, as in Bun et al.).
    Intuition: |...|^2 is precisely the factor by which the sample eigenvector
    mixes with noise, so each eigenvalue is shrunk by exactly the amount of its
    own unreliability.
    Valid for q < 1 (T > N).
    """
    lam = np.asarray(lam, float)
    n = lam.size
    if eta is None:
        eta = 1.0 / np.sqrt(n)
    z = lam - 1j * eta
    diff = z[:, None] - lam[None, :]                # (i, j): z_i - lam_j
    inv = 1.0 / diff
    np.fill_diagonal(inv, 0.0)                      # leave-one-out
    g = inv.sum(axis=1) / (n - 1)
    xi = lam / np.abs(1.0 - q + q * z * g) ** 2
    if preserve_trace and xi.sum() > 0:
        xi *= lam.sum() / xi.sum()
    return xi


def xi_oracle(u: np.ndarray, c_true: np.ndarray) -> np.ndarray:
    """The oracle: xi_i = u_i^T C u_i, the true variance along each *sample*
    eigenvector.  Uses the ground truth, so it is not an estimator — it is the
    Frobenius-optimal choice given sample eigenvectors.
    """
    return np.einsum("ij,jk,ki->i", u.T, c_true, u)


def clean_cov(
    x: np.ndarray,
    method: str = "rie",
    c_true: Optional[np.ndarray] = None,
    sigma2: float = 1.0,
) -> np.ndarray:
    """One-stop covariance estimate from returns x (N, T).

    method ∈ {"sample", "clip", "lw", "rie", "oracle"}.  "oracle" needs c_true.
    Eigen-based methods keep sample eigenvectors and replace eigenvalues.
    """
    x = np.asarray(x, float)
    n, t = x.shape
    q = n / t
    e = sample_cov(x)
    if method == "sample":
        return e
    if method == "lw":
        return lw_shrinkage(x)[0]
    lam, u = eigh_desc(e)
    if method == "clip":
        xi = xi_clip(lam, q, sigma2=sigma2)
    elif method == "rie":
        xi = xi_rie(lam, q)
    elif method == "oracle":
        if c_true is None:
            raise ValueError("oracle needs c_true")
        xi = xi_oracle(u, c_true)
    else:
        raise ValueError(f"unknown method {method!r}")
    return recompose(u, xi)


# --------------------------------------------------------------------------- #
#  Portfolio metrics (how wrongness turns into money)
# --------------------------------------------------------------------------- #
def min_var_weights(sigma: np.ndarray) -> np.ndarray:
    """Fully-invested minimum-variance weights  w ∝ Sigma^{-1} 1."""
    n = sigma.shape[0]
    w = np.linalg.solve(sigma, np.ones(n))
    return w / w.sum()


def portfolio_variance(w: np.ndarray, c: np.ndarray) -> float:
    """w^T C w — evaluate a portfolio on whichever matrix you claim is real."""
    return float(w @ c @ w)


def frobenius(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised Frobenius distance ||a - b||_F / sqrt(N) (per-asset scale)."""
    return float(np.linalg.norm(a - b, "fro") / np.sqrt(a.shape[0]))


def risk_report(sigma_est: np.ndarray, c_true: np.ndarray) -> dict:
    """The three numbers that summarise an estimator's honesty:

    - in_sample : the volatility the estimator *believes* its min-var portfolio has
    - realized  : the volatility that portfolio *actually* has under the truth
    - optimal   : the volatility of the true min-var portfolio (unreachable ideal)

    Perfect estimator: all three equal.  Sample covariance: believes (1-q) times
    the optimum, delivers 1/(1-q) times the optimum — wrong twice, in opposite
    directions.
    """
    w = min_var_weights(sigma_est)
    w_star = min_var_weights(c_true)
    return {
        "in_sample": np.sqrt(portfolio_variance(w, sigma_est)),
        "realized": np.sqrt(portfolio_variance(w, c_true)),
        "optimal": np.sqrt(portfolio_variance(w_star, c_true)),
    }


# --------------------------------------------------------------------------- #
#  N. Taleb-school instruments (fat-tail-safe measurement)
# --------------------------------------------------------------------------- #
def day_scale(x, robust: bool = True) -> np.ndarray:
    """Per-day cross-sectional scale, shape (1, T): divide by it to remove a
    common volatility mode.

    Each day's scale is estimated from the N *simultaneous* observations of that
    day — across the cross-section, never through time — so nothing here needs
    the time series of tail-dominated squares that breaks GARCH estimation.
    robust=True uses sqrt(pi/2) * MAD (MEAN absolute deviation, Taleb's MAD):
    a first-moment object whose influence is linear in an extreme rather than
    quadratic — tamer than std, though not bounded like the median version.
    The constant sqrt(pi/2) ~ 1.2533 makes it consistent for sigma on Gaussian
    cross-sections (E|x| = sigma*sqrt(2/pi)); when the cross-section itself is
    fat-tailed the constant is off — re-match it or trace-renormalise, since
    absolute-ruler devices (clip's MP edge, believed vols) inherit the units.
    robust=False uses the cross-sectional std — an L2 scale kept only for
    comparison, since a single wild asset can own it.
    """
    x = np.asarray(x, float)
    if robust:
        mean = np.mean(x, axis=0, keepdims=True)
        return np.sqrt(np.pi / 2.0) * np.mean(np.abs(x - mean), axis=0, keepdims=True)
    return x.std(axis=0, keepdims=True)


def max_to_sum_ratio(a, p: float = 4.0) -> np.ndarray:
    """Running R_n(p) = max_i |a_i|^p / sum_i |a_i|^p over the first n points.

    By the SLLN, R_n(p) -> 0 iff E|X|^p < infinity.  It is a moment-existence
    diagnostic: if R refuses to go to zero, the p-th moment is a fiction of
    your sample and every statistic built on it (kurtosis, GARCH fits, TW-edge
    reasoning) is measuring one observation.  
    """
    z = np.abs(np.ravel(np.asarray(a, float))) ** p
    return np.maximum.accumulate(z) / np.cumsum(z)


def kappa(sample_fn, n: int, n0: int = 1, reps: int = 20000,
          rng: Optional[np.random.Generator] = None) -> float:
    """Preasymptotic kappa(n0, n) = 2 - ln(n/n0) / ln(M(n)/M(n0)),
    where M(m) is the mean absolute deviation of an m-sum of iid draws from
    ``sample_fn(size, rng)``.

    kappa = 0: Gaussian-speed aggregation (M ~ sqrt(n)); kappa = 2 - alpha for
    alpha-stable.  Operationally: how much slower than sqrt(n) your errors
    shrink, i.e. how much more data the CLT actually costs you at this sample
    size.  MAD-based by construction — needs one moment, not four.
    """
    rng = np.random.default_rng() if rng is None else rng

    def mad_of_sum(m: int) -> float:
        s = sample_fn((reps, m), rng).sum(axis=1)
        return float(np.mean(np.abs(s - s.mean())))

    return 2.0 - np.log(n / n0) / np.log(mad_of_sum(n) / mad_of_sum(n0))


def _norm_ppf(u):
    """Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9).
    Pure numpy so the module keeps its no-scipy promise."""
    u = np.asarray(u, float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.empty_like(u)
    lo = u < plow
    hi = u > phigh
    mid = ~(lo | hi)
    if lo.any():
        q = np.sqrt(-2 * np.log(u[lo]))
        x[lo] = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                 / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - u[hi]))
        x[hi] = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                  / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if mid.any():
        q = u[mid] - 0.5
        r = q * q
        x[mid] = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                  / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))
    return x


def normal_scores(x) -> np.ndarray:
    """Rank-Gauss (van der Waerden) transform, per row: each asset's history ->
    ranks -> Gaussian quantiles.

    Destroys the marginal tails (every entry bounded by ~ Phi^{-1}(T/(T+1)))
    while keeping the dependence ordering — the robust way to feed fat-tailed
    data to correlation machinery.  Note what it does NOT do: a common
    volatility day still pushes *all* assets' ranks up together, so
    cross-sectional commonality survives (tamed, not removed). 
    """
    x = np.asarray(x, float)
    t = x.shape[1]
    r = np.argsort(np.argsort(x, axis=1), axis=1) + 1.0
    return _norm_ppf(r / (t + 1.0))
