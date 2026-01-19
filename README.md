# Vuong-Style Model Selection Tests in Python

This repository provides a **statsmodels-friendly Python implementation** of the three major families of Vuong-style non-nested model selection tests:

1. **Classical [Vuong (1989)](https://www.jstor.org/stable/1912557)**: studentized log-likelihood ratio with fixed normal critical values  
2. **[Shi (2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE382)**: *non-degenerate* modified Vuong procedure with simulation-based calibration (and optional adaptive tuning)  
3. **[Schennach–Wilhelm (2017)](https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1224716)**: *split-sample regularized* statistic, plus **bootstrap calibration** (including a **pairwise bootstrap** that preserves the split structure)

The repo also includes **Monte Carlo replication notebooks** (for my job market paper) that diagnose finite-sample miscalibration and show how split-sample regularization and bootstrap critical values can improve performance.

Paper: https://drive.google.com/file/d/14FdLzfvJzOyyH0F6itTg2TeE7dgiF9Jd/view

---

## Quickstart (statsmodels / `GenericLikelihoodModel`)

All tests take the same core objects, in the same order:

`(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2, ...)`

- `ll*`: per-observation log-likelihood contributions, shape `(n,)`
- `grad*`: per-observation score vectors, shape `(n, p)`
- `hess*`: full-sample Hessian, shape `(p, p)`
- `params*`: fitted parameters, shape `(p,)`

Example:

```python
from vuong_test_base import regular_test, ndVuong
from vuongtests11 import sw_test, sw_bs_test

# Fit model 1
m1 = your_model_1(y, X)
fit1 = m1.fit()
ll1     = m1.loglikeobs(fit1.params)
grad1   = m1.score_obs(fit1.params)
hess1   = m1.hessian(fit1.params)
params1 = fit1.params

# Fit model 2
m2 = your_model_2(y, X)
fit2 = m2.fit()
ll2     = m2.loglikeobs(fit2.params)
grad2   = m2.score_obs(fit2.params)
hess2   = m2.hessian(fit2.params)
params2 = fit2.params

# Classical Vuong (normal critical values)
decision = regular_test(ll1,grad1,hess1,params1, ll2,grad2,hess2,params2, alpha=.05)

# Shi (2015) non-degenerate Vuong
decision = ndVuong(ll1,grad1,hess1,params1, ll2,grad2,hess2,params2, alpha=.05, nsims=1000)

# Schennach–Wilhelm (2017) regularized statistic
decision = sw_test(ll1,grad1,hess1,params1, ll2,grad2,hess2,params2, epsilon=.5, alpha=.05)

# Bootstrap-calibrated S–W test (pairwise bootstrap recommended)
decision = sw_bs_test(ll1,grad1,hess1,params1, ll2,grad2,hess2,params2,
                      trials=500, epsilon=.5, alpha=.05, pairwise=True, seed=123)
```

Return value convention:
- `0`: no selection
- `1`: select model 1
- `2`: select model 2

---

## What’s implemented

### `vuong_test_base.py` (classical tests + Shi benchmark + bias-correction helpers)
- `regular_test(...)` — classical Vuong-style test (fixed normal critical values), optional bias correction  
- `two_step_test(...)` — two-stage benchmark (simulated stage-1 cutoff + normal decision rule)  
- `ndVuong(...)` — Shi (2015) non-degenerate modified Vuong procedure (simulation-calibrated)  
- Bias-correction utilities used for trace/eigenvalue-style adjustments (relevant for misspecification-induced numerator bias)

### `vuongtests11.py` (S–W regularization + bootstrap calibration)
- `sw_test(...)` — Schennach–Wilhelm (2017) split-sample regularized statistic (fixed normal critical values)  
- `sw_bs_test(...)` — bootstrap-calibrated S–W test  
  - `pairwise=False`: naive i.i.d. bootstrap  
  - `pairwise=True`: **pairwise bootstrap** preserving the even/odd split-sample structure

---

## Monte Carlo designs + replication notebooks (`revision_2025/`)

The replication workflow is organized as **Jupyter notebooks (`.ipynb`)** under `revision_2025/`. The notebooks cover two standard simulation designs used to stress-test Vuong-style procedures in finite samples:

### Example 1: random denominator (non-normal finite-sample tails)
Following the simulation frameworks in [Shi (2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE382) and [Schennach and Wilhelm (2017)](https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1224716), this design makes the variance of the per-observation log-likelihood difference occasionally very small. As a result, the usual studentized statistic can be skewed/heavy-tailed in finite samples, and fixed normal critical values can be badly miscalibrated (often extremely conservative).  
**Used for:** size tables, power curves, and refinement diagnostics (sampling vs bootstrap vs normal).

Notebooks:
- `revision_2025/sw_table1.ipynb` — Example 1 size results (null rejection frequencies)  
- `revision_2025/sw_table_abc.ipynb` — Example 1 results focused mostly on power  
- `revision_2025/refinement_denom_ex.ipynb` — Example 1 refinement/appendix figures (distribution-shape diagnostics; density overlays / tail behavior)

### Example 2: numerator bias (finite-sample bias under misspecification)
This design emphasizes finite-sample bias in the plug-in log-likelihood difference under misspecification (a Takeuchi-style phenomenon emphasized in [Shi (2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE382)). The bias can be especially visible with many weak regressors, motivating an **O(1/n)**-style trace/eigenvalue correction.  
**Used for:** size curves/tables under the null (often highlighting small/moderate n) and refinement diagnostics.

Notebooks:
- `revision_2025/sw_Table1_refinement_ex.ipynb` — Example 2 refinement evidence formatted as a table  
- `revision_2025/sw_table_d.ipynb` — Example 2 size results (multiple variants highlighting numerator bias)  
- `revision_2025/refinement_num_ex.ipynb` — Example 2 refinement/appendix figures (includes density overlays)

---

## Tuning and options (what matters in practice)

- `biascorrect=True`: applies a score/Hessian-based correction to the numerator (most relevant in Example 2 settings)
- `epsilon`: regularization strength in the S–W statistic (central in Example 1 / random-denominator settings)
- Bootstrap settings:
  - `trials`: bootstrap replications
  - `pairwise=True` recommended for S–W bootstrap calibration

---

## References

- Vuong (1989): https://www.jstor.org/stable/1912557  
- Shi (2015): https://onlinelibrary.wiley.com/doi/abs/10.3982/QE382  
- Schennach and Wilhelm (2017): https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1224716  

Older (archived/out-of-date) repo with empirical applications from earlier versions of the project:  
https://github.com/ericschulman/testing_empirical_ex

---
