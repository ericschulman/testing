## Overview

This repo contains code used to run the **Monte Carlo experiments** in [my job market paper](https://drive.google.com/file/d/14FdLzfvJzOyyH0F6itTg2TeE7dgiF9Jd/view). The experiments study and compare model selection tests in the spirit of [Vuong (1989)](https://www.jstor.org/stable/1912557), with a focus on improving finite-sample calibration using a **split-sample / pairwise bootstrap**. The two main Monte Carlo designs (Examples 1 and 2) are drawn from the simulation frameworks in [Shi (2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE382) and [Schennach and Wilhelm (2017)](https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1224716).

The code is designed to work naturally with [`GenericLikelihoodModel`](https://www.statsmodels.org/dev/dev/generated/statsmodels.base.model.GenericLikelihoodModel.html) in [`statsmodels`](https://www.statsmodels.org/stable/index.html).

---

## Background on the Monte Carlo designs (Example 1 vs Example 2)

**Example 1: random denominator (non-normal finite-sample tails)**  
Follows [Shi (2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE382) and [Schennach and Wilhelm (2017)](https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1224716). The design makes the variance of the per-observation log-likelihood difference sometimes very small, so the usual studentized statistic can be skewed/heavy-tailed in finite samples. Fixed normal critical values can then be miscalibrated (often extremely conservative).  
**Used for:** size tables, power curves, and “refinement” diagnostics (sampling vs bootstrap vs normal).

**Example 2: numerator bias (finite-sample bias under misspecification)**  
Emphasizes finite-sample bias in the plug-in log-likelihood difference under misspecification (a Takeuchi-style phenomenon emphasized in [Shi (2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE382)). This shows up strongly with many weak regressors, motivating an O(1/n)-style trace correction.  
**Used for:** size curves/tables under the null (often highlighting small/moderate n) and refinement diagnostics.

---

## Repo layout (Monte Carlo notebooks + test code)

### Monte Carlo notebooks (`revision_2025/`)
The current replication workflow is organized as **Jupyter notebooks (`.ipynb`)** under `revision_2025/`.

**Example 1 notebooks**
- `revision_2025/sw_table1/` — Example 1 size results in table form (null rejection frequencies).
- `revision_2025/sw_table_abc/` — Example 1 results focused mostly on power.
- `revision_2025/refinement_denom_ex/` — Example 1 refinement/appendix figures (distribution-shape diagnostics; density overlays / tail behavior).

**Example 2 notebooks**
- `revision_2025/sw_Table1_refinement_ex/` — Example 2 refinement evidence formatted as a table (instead of a figure).
- `revision_2025/sw_table_D/` — Example 2 size results, with multiple variants to highlight finite-sample numerator bias.
- `revision_2025/reifnement_num_Ex/` — Example 2 refinement/appendix figures (includes the density overlay plots).

### Test implementations (two main python files)

The Monte Carlo notebooks call into two modules. All procedures take the same core likelihood objects in the same order:  
`(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2, ...)`  
and return `0` (no selection), `1` (select model 1), or `2` (select model 2).

#### `vuong_test_base.py` (classical tests + Shi benchmark + bias-correction helpers)

- **Classical Vuong-style test (“Normal”)**: Studentizes the (optionally bias-corrected) log-likelihood difference and uses fixed normal critical values.  
  `regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2, alpha=.05, c=0, refinement_test=False, biascorrect=False, print_stuff=True)`

- **Two-step benchmark**: Two-stage classical procedure: a simulated stage-1 cutoff (via `compute_stage1`) followed by the usual normal-critical-values decision rule.  
  `two_step_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2, alpha=.05, biascorrect=False)`

- **Shi (2015) benchmark** ([Shi (2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE382)): Non-degenerate modified Vuong procedure with simulation-based calibration and an optional adaptive tuning step.  
  `ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2, alpha=.05, nsims=1000, adapt_c=True)`

- **Bias-correction helpers (Example 2)**: Score/Hessian-based utilities used to construct the trace/eigenvalue-style adjustment (e.g. `compute_eigen2(...)`). These are activated through `biascorrect=True` in the tests above.

#### `vuongtests11.py` (S–W regularization + bootstrap calibration)

This file focuses on the Schennach–Wilhelm-style regularized statistic and bootstrap calibration:

- **S–W regularized test (fixed normal critical values)** ([Schennach and Wilhelm (2017)](https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1224716)): Stabilizes the statistic using split-sample regularization controlled by `epsilon`, targeting the Example 1 “random denominator” issue.  
  `sw_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2, epsilon=.5, alpha=.05, biascorrect=False, print_stuff=False)`

- **Bootstrap-calibrated S–W test**: Uses bootstrap critical values for the S–W statistic. Set `pairwise=False` for the naive i.i.d. bootstrap and `pairwise=True` for the pairwise bootstrap that preserves the even/odd split structure (the main bootstrap method studied here).  
  `sw_bs_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2, alpha=.05, trials=500, epsilon=0.5, biascorrect=False, seed=None, print_stuff=False, pairwise=False)`

---

## Running the tests (shared inputs)

All tests in the Monte Carlo notebooks are called with the same core objects:

`(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2, ...)`

- `ll*`: per-observation log-likelihood contributions  
- `grad*`: per-observation score vectors  
- `hess*`: full-sample Hessians  
- `params*`: fitted parameters  

A typical pattern (for subclasses of [`GenericLikelihoodModel`](https://www.statsmodels.org/dev/dev/generated/statsmodels.base.model.GenericLikelihoodModel.html)):

```python
model1 = your_model(y, X)
fit1 = model1.fit()

ll1     = model1.loglikeobs(fit1.params)   # (n,)
grad1   = model1.score_obs(fit1.params)    # (n, p)
hess1   = model1.hessian(fit1.params)      # (p, p)
params1 = fit1.params
```

Most notebooks define a helper (often named `setup_test(...)` / `setup_shi(...)`) that fits both models and returns these inputs for model 1 and model 2 in the expected order.

---

## Tuning parameters and bias correction (what matters for the Monte Carlo results)

### Bias correction (`biascorrect`)
`biascorrect` appears in both the classical code path and the S–W/bootstrap code path. It toggles whether the numerator uses a trace/eigenvalue-style correction term computed from gradients/Hessians.

Practical guidance for the two Monte Carlo designs:
- **Example 2** is where `biascorrect=True` matters most (it targets numerator bias under misspecification).
- **Example 1** is where calibration choices matter most (normal vs bootstrap critical values; naive vs pairwise bootstrap).

### S–W tuning parameter (`epsilon`)
The S–W regularized statistic depends on `epsilon`, which controls how strongly the split-sample regularization enters the statistic. This is central in Example 1 (random denominator).

### “Optimal epsilon” (experimental)
`vuongtests11.py` contains experimental code to choose `epsilon` automatically (a data-driven tuning rule; see `compute_optimal_epsilon(...)`). Some notebooks use this as a sensitivity check/exploratory extension to see how performance changes with the degree of regularization.

---

## Notes on versions / branches (and older applications)

- The **current 2025 revision workflow** is centered on `vuong_test_base.py`, `vuongtests11.py`, and the notebooks under `revision_2025/`.
- Older `vuong_tests*` / `vuong_plots*` files are retained mainly for backward compatibility with older experiments and typically live on the **development** branch as part of the project’s iteration history.

Note: I also link to an older repo with empirical applications from earlier versions of the project:  
https://github.com/ericschulman/testing_empirical_ex  
That repo is **out of date** relative to this one and should be viewed mainly as an archive of older applications, not as the current implementation.