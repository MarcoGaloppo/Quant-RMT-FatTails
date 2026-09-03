# Eigenvalues in the wild — a random-matrix laboratory

Let us start frome the basic. If I have not misunderstood a lot: essentially every portfolio decision 
eventually runs through a covariance matrix. However, every covariance matrix gets estimated from 
too little data. **Numquam gaudium** as those wise latins would say. Indeed, with `N`assets and `T`
observations the natural parameter is

    q = N / T,

and in practice `q` just refuses to be small. For example, if we have about 500 names returns and two years 
of daily data, we get `q ≈ 1`. Now, one may ask: can't we just increase `T`? Only up to a point. 
Mixing time periods in finance can create problems sometimes (think of how many successfull 
business have sprouted only in the last 10 years!)

Now, it is known that the sample covariance `E = X Xᵀ / T` is the maximum-likelihood estimator (MLE),
for the population covariance, unbiased entry by entry. Yet, its **eigenvalues are systematically,                             wrong**. In other words, not great. Thankfully, Random Matrix Theory (RMT) can helps us out.

In this small project we will try to apply RMT to finance data after looking into some simulations
which will serve as guidance and exercises.

## The set-up

- `rmt_lab.py` — pure-numpy module: ensemble generators (Wigner/Wishart with
  Gaussian, uniform, Rademacher, Student-t, α-stable and elliptical entries),
  the theoretical laws (semicircle, Marchenko–Pastur, BBP spike formulas), the
  covariance estimators we race (eigenvalue clipping, Ledoit–Wolf linear
  shrinkage, RIE / nonlinear shrinkage, and the ground-truth oracle), and the
  Taleb-school instruments (MAD day-scales, max-to-sum moment diagnostics, the
  κ preasymptotic metric, rank-Gauss robust correlations). 
- `narrative_rmt_simulations.ipynb` — our laboratory. We directly employ only
  synthetic data where the true covariance is known. Hence, every claim is checkable
  and every estimator can be scored against the truth. 
- `narrative_rmt_market_data.ipynb` —  where we follow Alice down the white rabbit's hole.
  We now employ S&P constituents and futures to ask a few questions. For example: does the bulk
  match Marchenko–Pastur, which outliers are real, and does the cleaning ranking established
  in simulation survive out-of-sample reality. My first attempt at becoming a practitioner.

## Simulation notebook

**Part 0** contains the opening gambit: at `q = 0.5` the in-sample volatility of a
sample-covariance min-variance portfolio understates its true volatility by a
factor `(1-q)`. To wit, one believes 5% of risk and but carries a whooping 10%. 
And then the firm goes bust! 

**Parts 1–2** build the two null models. Wigner's semicircle: the eigenvalues of
pure symmetric noise are not "spread randomly", they converge to a deterministic
law. Marchenko–Pastur: sample covariances of *uncorrelated* data occupy a deterministic bulk
`[σ²(1−√q)², σ²(1+√q)²]`. Note, that essentially MP is the null hypothesis of correlation-matrix 
mining. We then explore Tracy–Widom and past it into the **large-deviation regime**: how does the probability
that noise gives us an eigenvalue a *finite* distance beyond the edge decay? 

**Part 3** is universality: replace Gaussian entries by uniform, by ±1 coin flips,
by skewed centred exponentials and let us watch that beautiful spectra do not move. 
The semicircle really is the free CLT of matrices, and MP also carries universality.
Just awesome, if you ask me. However, something which is kind of like the CLT should
beware the role of moments...

**Parts 4–5** fat tails enter, stage left. Student-t entries: the emperical spectral 
density (ESD) *bulk* survives down to ν > 2 (hell yeah), but for ν < 4 the *edge* breaks (hell nah).
To wit, the largest eigenvalue detaches, grows with N, and its eigenvector localises on a single
entry. A single fat-tailed day manufactures a **fake factor**. Then an over-simplified market's
version: elliptical returns (one common volatility for all assets each day). Just this already
deforms the bulk itself even after per-asset standardisation. Finally, a short α-stable coda marks 
the boundary where the semicircle law itself dies.

**Part 6** plants true factors in the covariance and goes to look for the BBP phase
transition: a factor of strength `ℓ` produces a visible outlier only if `ℓ > 1 + √q`. 
Below the threshold it is invisible. Literally, it is *impossible* to detect it.
Above the threshold instead, the outlier's position overstates `ℓ` and its eigenvector 
is only partially aligned with the truth. To wit, **detected ≠ estimated**.

**Part 7** is the final shoot-out. Because the truth is known, we can do "supervised" tests
for the estimators. Raw sample vs eigenvalue clipping vs Ledoit–Wolf vs RIE vs the oracle,
under Gaussian returns and under elliptical t(3) returns. 

**Part 8** simply gives the checklist the data notebook will execute.

## Headline results of the simulation laboratory 

1. **In-sample risk is a lie. Who could have guessed?** The sample
   min-variance portfolio's in-sample variance understates its true variance by
   `(1-q)²` — a factor 4 at `q = 1/2`. 
2. **Pure noise has structure.** An eigenvalue 1.8 from a correlation matrix at
   `q = 0.25` is not evidence of anything. 
3. **Universality holds — with a 4th-moment clause.** Coin-flip entries give the
   same spectrum as Gaussians. That is really cool! Student-t(3) does too *in the bulk*,
   but its edge grows fake factors whose eigenvectors are localised. To with, with fat tails,
   the top of the spectrum is where one is fooled by randomness.
4. **Common volatility bends the bulk.** Elliptical t returns violate MP with zero
   true correlation structure — a caution for every "eigenvalue above the MP edge
   = signal" argument applied to raw returns. Of course, returns are not really elliptical 
   (see e.g., Chicheportiche & Bouchaud (2012))
5. **Weak factors are invisible, strong ones are biased.** BBP threshold
   `1 + √q`; above it, overlap < 1 and position inflated by `qℓ/(ℓ−1)`.
6. **Cleaning works!** The raw sample carries +30% excess portfolio risk at `q = 1/2`. 
   Instead, every cleaned estimator lands within a few ercent of the oracle ceiling. 
   RIE tracks the oracle on *both* losses because it shrinks each eigenvalue by exactly the 
   overlap-weighted amount BBP says is lost. Interestingly, Ledoit–Wolf,
   which ain't great in Frobenius, wins the min-variance race. Under elliptical fat tails the
   spectrum-reading methods get miscalibrated and volatility filtering restores
   them. It cannot do so under fat-tailed entries.

## Running it

    pip install -r requirements.txt
    jupyter lab narrative_rmt_simulations.ipynb

Everything is seeded; the notebook runs top-to-bottom in a couple of minutes.
`rmt_lab.py` needs only numpy.

## References

- Marchenko & Pastur (1967), Mathematics of the USSR-Sbornik 1, 457 - 483 — *Distribution of eigenvalues for some sets of random  matrices*
- Laloux, Cizeau, Bouchaud, and Potters (1999), PRL 83, 1467 — *Noise dressing of financial correlation matrices*
- Plerou et al. (1999), PRL 83, 1471 — *Universal and nonuniversal properties of cross correlations in financial time series*
- Baik & Péché (2005), Ann. Probab. 33, 1643-1697 — *Phase transition of the largest eigenvalue for nonnull complex  sample covariance matrices*
- Dean & Majumdar (2006), PRL 97, 160201 — *Large Deviations of Extreme Eigenvalues of Random Matrices*
- Majumdar & Vergassola (2009), PRL 102, 060601 — *Large Deviations of the Maximum Eigenvalue for Wishart and Gaussian Random Matrices*
- Majumdar & Schehr (2014), J. Stat. Mech. P01012  — *Top eigenvalue of a random matrix: large deviations and third order phase transition*
- Ledoit & Wolf (2004), JMVA 88, 365-411 — *A well-conditioned estimator for large-dimensional covariance matrices*
- Ledoit & Péché (2011), Probab. Theory Relat. Fields 151, 233–264 — *Eigenvectors of some large sample covariance matrix ensembles*
- Bun, Bouchaud, and Potters (2017), Phys. Rep. 666, 1 - 109 — *Cleaning large correlation matrices: tools from RMT* (namely, the review this lab shadows)
- Cizeau & Bouchaud (1994), PRE 50, 1810 — *Theory of Lévy matrices*
- Chicheportiche & Bouchaud (2012), IJTAF 15, 1250019 — *The joint distribution of stock returns is not elliptical* 
- Potters & Bouchaud (2020) — *A First Course in Random Matrix Theory*
- Taleb (2025) - *Statistical Consequences of Fat Tails*

Author: Marco Galoppo
