"""Tests for rmt_lab.

Here we run two kind of test:

1. **Invariants** — statements with an exact right answer (normalisations, algebraic
   inverses, trace preservation). 
2. **Seeded statistical checks** — Monte Carlo against the theory we want to implement 
   (BBP positions and overlaps, the cleaning ordering, kappa). These use a
   fixed seed and account for tolerances.

Run with:  pytest -q
"""
import numpy as np
import pytest

# np.trapezoid is numpy>=2.0; np.trapz is the older spelling
trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

from rmt_lab import (
    sample_entries, sample_stable, wigner, goe, returns_gaussian, returns_student,
    sample_cov, sample_corr, semicircle_pdf, mp_edges, mp_pdf, stieltjes,
    spiked_cov, bbp_threshold, spike_lambda, spike_overlap2, spike_ell_from_lambda,
    eigh_desc, ipr, recompose, xi_clip, lw_shrinkage, xi_rie, xi_oracle, clean_cov,
    min_var_weights, portfolio_variance, frobenius, risk_report,
    day_scale, max_to_sum_ratio, kappa, normal_scores,
)

SEED = 20260901


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
#  1. Invariants
# --------------------------------------------------------------------------- #
def test_semicircle_normalised():
    x = np.linspace(-2.5, 2.5, 20_001)
    assert trapz(semicircle_pdf(x), x) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.9])
def test_mp_density_normalised(q):
    lo, hi = mp_edges(q)
    x = np.linspace(lo * 0.5, hi * 1.1, 40_001)
    assert trapz(mp_pdf(x, q), x) == pytest.approx(1.0, abs=3e-3)


def test_mp_density_mass_is_one_over_q_when_q_above_one():
    """For q > 1 the continuous part carries 1/q; the rest is a delta at zero."""
    x = np.linspace(1e-4, 8.0, 40_001)
    assert trapz(mp_pdf(x, 2.0), x) == pytest.approx(0.5, abs=5e-3)


def test_mp_edges_symmetry_and_scaling():
    lo, hi = mp_edges(0.25, sigma2=3.0)
    assert (lo, hi) == pytest.approx((3.0 * 0.25, 3.0 * 2.25))


@pytest.mark.parametrize("ell", [1.8, 2.5, 5.0, 50.0])
@pytest.mark.parametrize("q", [0.1, 0.4])
def test_bbp_inverse_round_trip(ell, q):
    """spike_ell_from_lambda must invert spike_lambda exactly above threshold."""
    assert ell > bbp_threshold(q)
    lam = float(spike_lambda(ell, q))
    assert float(spike_ell_from_lambda(lam, q)) == pytest.approx(ell, rel=1e-8)


def test_bbp_below_threshold_is_flat_and_uninvertible():
    q = 0.25
    ell = bbp_threshold(q) * 0.9
    assert float(spike_lambda(ell, q)) == pytest.approx(mp_edges(q)[1])
    assert float(spike_overlap2(ell, q)) == 0.0
    assert np.isnan(spike_ell_from_lambda(1.2, q))          # inside the bulk


def test_overlap_increases_to_one():
    q = 0.25
    ov = np.array([float(spike_overlap2(e, q)) for e in (1.6, 2.0, 5.0, 100.0)])
    assert np.all(np.diff(ov) > 0)
    assert ov[-1] == pytest.approx(1.0, abs=1e-2)


def test_sample_corr_has_unit_diagonal(rng):
    C = sample_corr(returns_student(30, 200, 4.0, rng), demean=True)
    assert np.allclose(np.diag(C), 1.0)


def test_eigh_desc_is_descending_and_orthonormal(rng):
    lam, U = eigh_desc(sample_cov(returns_gaussian(40, 120, rng)))
    assert np.all(np.diff(lam) <= 1e-12)
    assert np.allclose(U.T @ U, np.eye(40), atol=1e-10)


def test_recompose_round_trip(rng):
    M = sample_cov(returns_gaussian(25, 90, rng))
    lam, U = eigh_desc(M)
    assert np.allclose(recompose(U, lam), M, atol=1e-10)


def test_ipr_bounds(rng):
    n = 200
    uniform = np.ones(n) / np.sqrt(n)
    assert ipr(uniform).item() == pytest.approx(1.0 / n)          # most spread out
    spike = np.zeros(n); spike[3] = 1.0
    assert ipr(spike).item() == pytest.approx(1.0)                 # one asset
    _, U = eigh_desc(goe(n, np.random.default_rng(1)))
    assert ipr(U).mean() == pytest.approx(3.0 / n, rel=0.15)       # Porter-Thomas


@pytest.mark.parametrize("method", ["clip", "rie"])
def test_eigenvalue_maps_preserve_trace(method, rng):
    lam, _ = eigh_desc(sample_cov(returns_gaussian(60, 200, rng)))
    xi = xi_clip(lam, 0.3) if method == "clip" else xi_rie(lam, 0.3)
    assert xi.sum() == pytest.approx(lam.sum(), rel=1e-8)
    assert np.all(xi > 0)


def test_min_var_weights_sum_to_one(rng):
    w = min_var_weights(sample_cov(returns_gaussian(30, 200, rng)))
    assert w.sum() == pytest.approx(1.0)


def test_day_scale_recovers_sigma_on_gaussian(rng):
    sigma = 1.7
    X = rng.standard_normal((300, 400)) * sigma
    assert np.mean(day_scale(X)) == pytest.approx(sigma, rel=0.02)


def test_day_scale_influence_is_linear_not_bounded(rng):
    """day_scale uses MAD, so a single wild asset moves it *linearly*. Namely, much tamer than std, but NOT ignored the way the
    median version would ignore it."""
    X = rng.standard_normal((200, 50))
    X[0] += 500.0                                          # one crazy name
    robust, l2 = day_scale(X).mean(), day_scale(X, robust=False).mean()
    assert l2 > 4 * robust                                 # far tamer than L2
    assert robust > 2.0                                    # but not immune
    med = 1.4826 * np.median(np.abs(X - np.median(X, axis=0, keepdims=True)),
                             axis=0).mean()
    assert med == pytest.approx(1.0, rel=0.15)             # median-AD *is* immune


def test_normal_scores_bounded_and_standardised(rng):
    Z = normal_scores(returns_student(20, 500, 3.0, rng, elliptical=False))
    assert Z.std() == pytest.approx(1.0, rel=0.05)
    assert np.abs(Z).max() < 4.0                       # marginals destroyed by design


def test_stieltjes_matches_definition(rng):
    lam, _ = eigh_desc(sample_cov(returns_gaussian(20, 80, rng)))
    z = 2.0 + 0.5j
    assert stieltjes(z, lam) == pytest.approx(np.mean(1.0 / (lam - z)))


# --------------------------------------------------------------------------- #
#  2. Seeded statistical checks
# --------------------------------------------------------------------------- #
def test_semicircle_matches_goe_spectrum():
    lam = np.linalg.eigvalsh(goe(1500, np.random.default_rng(SEED)))
    assert lam.var() == pytest.approx(1.0, abs=0.05)          # radius-2 semicircle
    assert lam.max() == pytest.approx(2.0, abs=0.12)


def test_mp_edges_match_a_wishart_spectrum():
    n, t = 400, 1600
    lam = np.linalg.eigvalsh(sample_cov(returns_gaussian(n, t, np.random.default_rng(SEED))))
    lo, hi = mp_edges(n / t)
    assert lam.mean() == pytest.approx(1.0, abs=0.03)
    assert lam.max() == pytest.approx(hi, abs=0.12)
    assert lam.min() == pytest.approx(lo, abs=0.08)


def test_bbp_position_and_overlap_match_simulation():
    """The headline BBP predictions, measured."""
    rng = np.random.default_rng(SEED)
    n, t, ell = 400, 1600, 4.0
    q = n / t
    C, V = spiked_cov(n, [ell], rng)
    lams, ovs = [], []
    for _ in range(8):
        lam, U = eigh_desc(sample_cov(returns_gaussian(n, t, rng, cov=C)))
        lams.append(lam[0]); ovs.append((U[:, 0] @ V[:, 0]) ** 2)
    assert np.mean(lams) == pytest.approx(float(spike_lambda(ell, q)), rel=0.03)
    assert np.mean(ovs) == pytest.approx(float(spike_overlap2(ell, q)), abs=0.03)


def test_rie_tracks_oracle():
    rng = np.random.default_rng(SEED)
    n, t = 250, 500
    C, _ = spiked_cov(n, [10.0, 4.0, 2.0], rng)
    X = returns_gaussian(n, t, rng, cov=C)
    e_rie = frobenius(clean_cov(X, "rie", c_true=C), C)
    e_orc = frobenius(clean_cov(X, "oracle", c_true=C), C)
    e_raw = frobenius(clean_cov(X, "sample"), C)
    assert e_rie < e_raw                                   # cleaning helps at all
    assert e_rie / e_orc < 1.4                             # and lands near the ceiling


def test_cleaning_beats_raw_sample_out_of_sample():
    rng = np.random.default_rng(SEED)
    n, t = 200, 400
    C, _ = spiked_cov(n, [8.0, 3.0], rng)
    risks = {m: [] for m in ("sample", "clip", "lw", "rie")}
    for _ in range(6):
        X = returns_gaussian(n, t, rng, cov=C)
        for m in risks:
            risks[m].append(risk_report(clean_cov(X, m, c_true=C), C)["realized"])
    mean = {m: np.mean(v) for m, v in risks.items()}
    for m in ("clip", "lw", "rie"):
        assert mean[m] < mean["sample"]


def test_in_sample_risk_understates_by_one_minus_q():
    """Part 0 of the simulation notebook, as an assertion."""
    rng = np.random.default_rng(SEED)
    n, q = 250, 0.5
    reps = [risk_report(sample_cov(returns_gaussian(n, int(n / q), rng)), np.eye(n))
            for _ in range(15)]
    believed = np.mean([r["in_sample"] for r in reps])
    realized = np.mean([r["realized"] for r in reps])
    assert believed / realized == pytest.approx(1 - q, rel=0.05)


def test_kappa_approaches_two_minus_alpha_and_is_monotone():
    """kappa(1, 30) is a *preasymptotic* metric, so it sits below the asymptotic
    2 - alpha and the shortfall grows as the tails get heavier (0.66 vs 0.8 at
    alpha = 1.2). Monotonicity in alpha is the robust property."""
    ks = [kappa(lambda size, g, a=a: sample_stable(a, size, g), 30,
                rng=np.random.default_rng(SEED)) for a in (1.2, 1.5, 1.8)]
    assert ks[0] > ks[1] > ks[2] > 0.0                     # heavier tails, larger kappa
    assert ks[1] == pytest.approx(0.5, abs=0.12)           # 2 - alpha at alpha = 1.5
    assert all(k < 2.0 - a for k, a in zip(ks, (1.2, 1.5, 1.8)))   # always below


def test_kappa_of_gaussian_is_zero():
    k = kappa(lambda size, g: g.standard_normal(size), 30,
              rng=np.random.default_rng(SEED))
    assert k == pytest.approx(0.0, abs=0.05)


def test_max_to_sum_separates_finite_from_infinite_moments():
    rng = np.random.default_rng(SEED)
    n = 200_000
    gauss = max_to_sum_ratio(rng.standard_normal(n), 4)[-1]
    heavy = max_to_sum_ratio(rng.permutation(rng.standard_t(2.5, n)), 4)[-1]
    assert gauss < 0.01                                    # fourth moment exists
    assert heavy > 0.05                                    # it does not


def test_fat_tailed_entries_break_the_edge_but_not_the_bulk():
    """Part 4 of the simulation notebook: nu < 4 detaches lambda_max."""
    rng = np.random.default_rng(SEED)
    n, t = 300, 900
    edge = mp_edges(n / t)[1]
    thin = np.linalg.eigvalsh(sample_cov(returns_student(n, t, 8.0, rng, elliptical=False)))
    fat = np.linalg.eigvalsh(sample_cov(returns_student(n, t, 2.5, rng, elliptical=False)))
    assert thin.max() < 1.15 * edge                        # pinned to the edge
    assert fat.max() > 1.5 * edge                          # detached
    assert np.median(fat) == pytest.approx(np.median(thin), rel=0.25)   # bulk survives
