# Eigenvalues in the wild — a random-matrix laboratory that eventually meets the market

[![tests](https://github.com/MarcoGaloppo/Quant-RMT-FatTails/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcoGaloppo/Quant-RMT-FatTails/actions/workflows/ci.yml)

Let us start from the basics. Essentially every portfolio decision eventually runs through a covariance matrix. However, every covariance matrix gets estimated from too little data. **Numquam gaudium** as those wise Latins would say. Indeed, with `N` assets and `T` observations the natural parameter is

    q = N / T,

and in practice `q` just refuses to be small. For example, if we have about 500 names returns and two years 
of daily data, we get `q ≈ 1`. Now, one may ask: can't we just increase `T`? Only up to a point. 
Mixing time periods in finance can create problems sometimes (think of how many successful 
businesses have sprouted only in the last 10 years!).

Now, it is known that the sample covariance `E = X Xᵀ / T` is the maximum-likelihood estimator (MLE),
for the population covariance, unbiased entry by entry. Yet, its **eigenvalues are systematically wrong**. In other words, not great. Thankfully, Random Matrix Theory (RMT) can help us out.

In this project we will apply RMT to finance data after looking into some simulations which will serve as guidance and exercises.

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
  in simulation survive out-of-sample reality.
- `tests/test_rmt_lab.py` — 38 checks on the module, run by CI on every push. Two kinds:
  invariants with exact answers (normalisations, the BBP inverse round-trip, trace
  preservation) and seeded statistical checks against the theory the module implements
  (BBP positions and overlaps, the cleaning ordering, κ).

## Simulation notebook

**Part 0** contains the opening gambit: at `q = 0.5` the in-sample volatility of a
sample-covariance min-variance portfolio understates its true volatility by a
factor `(1-q)`. To wit, one believes 5% of risk but carries a whopping 10%. 
And then the firm goes bust! 

**Parts 1–2** build the two null models. Wigner's semicircle: the eigenvalues of
pure symmetric noise are not "spread randomly", they converge to a deterministic
law. Marchenko–Pastur: sample covariances of *uncorrelated* data occupy a deterministic bulk
`[σ²(1−√q)², σ²(1+√q)²]`. Note that essentially MP is the null hypothesis of correlation-matrix 
mining. We then explore Tracy–Widom and past it into the **large-deviation regime**: how does the probability
that noise gives us an eigenvalue a *finite* distance beyond the edge decay? 

**Part 3** is universality: replace Gaussian entries by uniform, by ±1 coin flips,
by skewed centred exponentials, and let us watch — those beautiful spectra do not move. 
The semicircle really is the free CLT of matrices, and MP also carries universality.
Just awesome, if you ask me. However, something which is kind of like the CLT should
beware the role of moments...

**Parts 4–5** fat tails enter, stage left. Student-t entries: the empirical spectral 
density (ESD) *bulk* survives down to ν > 2 (good), but for ν < 4 the *edge* breaks (not good).
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
   but its edge grows fake factors whose eigenvectors are localised. To wit, with fat tails,
   the top of the spectrum is where one is fooled by randomness.
4. **Common volatility bends the bulk.** Elliptical t returns violate MP with zero
   true correlation structure — a caution for every "eigenvalue above the MP edge
   = signal" argument applied to raw returns. Of course, returns are not really elliptical 
   (see e.g., Chicheportiche & Bouchaud (2012)).
5. **Weak factors are invisible, strong ones are biased.** BBP threshold
   `1 + √q`; above it, overlap < 1 and position inflated by `qℓ/(ℓ−1)`.
6. **Cleaning works!** The raw sample carries +30% excess portfolio risk at `q = 1/2`. 
   Instead, every cleaned estimator lands within a few percent of the oracle ceiling. 
   RIE tracks the oracle on *both* losses because it shrinks each eigenvalue by exactly the 
   overlap-weighted amount BBP says is lost. Interestingly, Ledoit–Wolf,
   which is not great in Frobenius, wins the min-variance race. Under elliptical fat tails the
   spectrum-reading methods get miscalibrated and volatility filtering restores
   them. It cannot do so under fat-tailed entries.

## Market notebook

**Part 0** is our data layer. We employ ~220 liquid S&P names tagged by sector. We work with log
returns by default and net returns where money is summed across assets. We set up a per-window
completeness rule instead of forward-filling, and, of course, **no winsorising anywhere, ever**. 
Since tails are important, this Part ends with a browsable tail viewer: pick a ticker, look at
`P(|r| > x)` on log-log axes against a same-variance Gaussian and a slope −3 guide.

**Part 1** is where we sample a SCM on a two-year window (`N = 216`, `T = 501`, `q = 0.43`). We then
test the ESD against MP using (i) the naive edge, (ii) the effective edge after removing the market mode, 
and (iii) TW plus the LDT cost. We then run a calibration control by doing the same experiment on synthetic data with **four** planted factors. We use both Gaussian and elliptical `t(3)`, to learn how to *read* whatever count the S&P gives us.

**Part 2** asks what the detected modes are made of. We produce eigenvector portraits coloured by sector, a
sector-concentration heatmap, gauge the obtained IPR against its `3/N` null, as well as the BBP overlap inverted mode by mode, and employ the Porter–Thomas test on 2024–26 data. Then we turn to the *bottom*
of the spectrum, which is where the danger actually lives.

**Part 3** is the volatility-filtering experiment. We test raw versus day-standardised data to see which outliers survive, which directions survive, and overall how much common volatility there is. We also look at the tails on **raw** returns only (statistics employed here: `R₂`, `R₄`, `κ`).

**Part 4** asks whether any of it persists for different time windows. We set up a rolling 252-day window stepped monthly, and we look at subspace overlaps rather than eigenvector overlaps. This is investigated with a lag — 12 months, namely the first with no shared days. The null is the delicate part and as such it gets four versions (Gaussian, elliptical `t(3)`, entries `t(3)`, both), with the same filter applied to both sides.

**Part 5** finally scores cleaning algorithms on real market data. We use rolling out-of-sample min-variance, a total of 175 rebalances, and score sample vs clipping vs Ledoit–Wolf vs RIE vs `1/N` vs filtered RIE. These are scored on realized volatility, **honesty** (believed at formation against realized over the hold), turnover and gross exposure. Plus a `q`-sweep that tests the theory's own prediction about when cleaning should stop being meaningful.

**Part 6** changes the playing field, moving to ~30 instruments across six asset classes at `q ≈ 0.01` (the backtest there runs on a 756-day window, so `q ≈ 0.04`). We look at the spectrum, the cross-asset blocks, the rolling equity–bond correlation, and a cleaning race in the regime where the machinery should have nothing left to do. 

**Part 7** closing statements and what comes next...

## Headline results of the market laboratory

1. **Most of the spectrum is noise, but thirteen eigenvalues stand out.** At `q = 0.43`, 203
   of 216 eigenvalues sit below the top of the MP band. Thirteen survive volatility filtering, and the
   top six can be identified with sectors: market, defensives-vs-growth, Energy, software, banks, and managed care.
2. **Detected ≠ estimated ≠ persistent.** BBP overlap falls from 0.99 to 0.57 across the
   thirteen, and twelve-month subspace persistence falls from 0.96 to 0.46 across ranks. Hence, the
   true count of tradeable structure is **only a handful, not thirteen**.
3. **The tails are cubic and half of them are idiosyncratic.** We have fat tails with an index of about 3. `R₄` never converges, and as such TW and LDT regarding the *edge* of MP does not apply. Day-standardising halves the fourth-moment concentration and no more: common volatility and single-name jumps in roughly equal measure. 
4. **The danger is at the bottom, not the top.** 36 eigenvalues remain below the lower MP edge
   holding 1% of the trace. Given that they are weighted by `1/λ` in every optimiser, they can be a real problem. Also, we note that these are made of business twins (DHI/LEN, MCO/SPGI, MPC/VLO). Repairing the smallest half of the spectrum removes 82% of the excess risk; the largest half only 14%.
5. **Cleaning pays — in a regime.** Looking at the realized volatility out-of-sample we found RIE 0.121, LW 0.125, clipping 0.127, sample 0.135, and `1/N` 0.163. The advantage scales with `q`, from +42% at `q = 0.8` to +3% at `q = 0.2`, and in the Part 6 backtest (`q ≈ 0.04`) it is essentially **zero** (clipping actively hurts).
6. **Nobody is honest, and it is not sampling noise.** Indeed, for RIE we found that the ratio believed/realized vol is 0.47–0.61 pooled, 0.77 by median, and below one in 84% of holds. Importantly, the gap does *not* close as `q → 0` the way stationary theory demands. Part 4 found the culprit in the eigenvectors: against a null calibrated to this market's own tails, real structure moves more than sampling noise permits. 
7. **The variance ranking does not survive tail scoring.** RIE wins on volatility, LW on
   expected shortfall, clipping on maximum drawdown. Squeezing the variance concentrates what remains into the tail. This notebook can say which estimator minimises variance; it cannot yet say which keeps you
   solvent. That is the next project.

## Running it

    pip install -r requirements.txt
    jupyter lab narrative_rmt_simulations.ipynb      # synthetic, seeded, a few minutes
    jupyter lab narrative_rmt_market_data.ipynb      # real data, ~30 s once cached

    pytest tests/ -q                                # 38 checks, ~1.5 s

The simulation notebook is fully seeded and needs no network. The market notebook downloads
prices once via `yfinance` into `data/` (gitignored) and runs offline from the cache
afterwards; `pyarrow` is worth installing so the cache is parquet rather than a
pandas-version-locked pickle. `rmt_lab.py` itself needs only numpy.

## Main References

- Marchenko & Pastur (1967), Mathematics of the USSR-Sbornik 1, 457 - 483 — *Distribution of eigenvalues for some sets of random matrices*
- Laloux, Cizeau, Bouchaud, and Potters (1999), PRL 83, 1467 — *Noise dressing of financial correlation matrices*
- Plerou et al. (1999), PRL 83, 1471 — *Universal and nonuniversal properties of cross correlations in financial time series*
- Baik & Péché (2005), Ann. Probab. 33, 1643-1697 — *Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices*
- Dean & Majumdar (2006), PRL 97, 160201 — *Large Deviations of Extreme Eigenvalues of Random Matrices*
- Majumdar & Vergassola (2009), PRL 102, 060601 — *Large Deviations of the Maximum Eigenvalue for Wishart and Gaussian Random Matrices*
- Majumdar & Schehr (2014), J. Stat. Mech. P01012 — *Top eigenvalue of a random matrix: large deviations and third order phase transition*
- Ledoit & Wolf (2004), JMVA 88, 365-411 — *A well-conditioned estimator for large-dimensional covariance matrices*
- Ledoit & Péché (2011), Probab. Theory Relat. Fields 151, 233–264 — *Eigenvectors of some large sample covariance matrix ensembles*
- Bun, Bouchaud, and Potters (2017), Phys. Rep. 666, 1 - 109 — *Cleaning large correlation matrices: tools from RMT* (namely, the review this lab shadows)
- Cizeau & Bouchaud (1994), PRE 50, 1810 — *Theory of Lévy matrices*
- Chicheportiche & Bouchaud (2012), IJTAF 15, 1250019 — *The joint distribution of stock returns is not elliptical* 
- Potters & Bouchaud (2020) — *A First Course in Random Matrix Theory*
- Taleb (2025) — *Statistical Consequences of Fat Tails*

Additionally used in the market notebook:

- Porter & Thomas (1956), Phys. Rev. 104, 483 — *Fluctuations of nuclear reaction widths* 
- Plerou et al. (2002), PRE 65, 066126 — *Random matrix approach to cross correlations in financial data* 
- Pafka & Kondor (2003), Physica A 319, 487-494 — *Noisy covariance matrices and portfolio optimization II*
- El Karoui (2010), Ann. Statist. 38, 3487-3566 — *High-dimensionality effects in the Markowitz problem and other quadratic programs with linear constraints: risk underestimation*
- DeMiguel, Garlappi, and Uppal (2009), Rev. Financ. Stud. 22, 1915-1953 — *Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy?*
- Bun, Bouchaud, and Potters (2018), PRE 98, 052145 — *Overlaps between eigenvectors of correlated random matrices*
- Tumminello, Lillo, and Mantegna (2007), EPL 78, 30006 — *Hierarchically nested factor model from multivariate data*
- Bongiorno & Challet (2021), PLOS ONE 16(1), e0245092 — *Covariance matrix filtering with bootstrapped hierarchies* 
- Bongiorno, Challet, and Loeper (2021), arXiv:2111.13109 — *Cleaning the covariance matrix of strongly nonstationary systems with time-independent eigenvalues* (Route 3: the transient-mode fix)
- Bongiorno & Challet (2022), arXiv:2112.07521 — *Non-linear shrinkage of the price return covariance matrix is far from optimal for portfolio optimisation*
- Bouchaud, Mastromatteo, Potters, and Tikhonov (2022), arXiv:2205.01012 — *Excess out-of-sample risk and fleeting modes*
- Karami, Benichou, Benzaquen, and Bouchaud (2021), Wilmott 111, 63-73 — *Conditional correlations and principal regression analysis for futures*

Author: Marco Galoppo
