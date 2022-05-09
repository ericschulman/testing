# Overview

This repo contains experimental code used to develop the theory in [my job market paper](https://drive.google.com/file/d/14FdLzfvJzOyyH0F6itTg2TeE7dgiF9Jd/view). The repository focuses on improving the [Vuong (1989)](https://www.jstor.org/stable/1912557) test with the bootstrap. The code is designed to be compatible with [`GenericLikelihoodModel`](https://www.statsmodels.org/dev/dev/generated/statsmodels.base.model.GenericLikelihoodModel.html) class in [`statsmodels`](https://www.statsmodels.org/stable/index.html).

For a less experimental version of the selection tests see [the following repository](https://github.com/ericschulman/testing_empirical_ex).

# Main modules provided

There are multiple versions of `vuong_tests` and `vuong_plots` in this repository. These modules contain the main code for running the test. I included the old versions so that the older examples would still be able to run.  `statsmodels` automatically compute these things with the [`GenericLikelihoodModel`](https://www.statsmodels.org/dev/dev/generated/statsmodels.base.model.GenericLikelihoodModel.html) class in [`statsmodels`](https://www.statsmodels.org/stable/index.html). .

The main versions of the test used in the paper are in `vuong_tests4.py`. `vuong_tests5.py`  contains experimental code to avoid needing to compute the likelihood on each observation and other experimental bootstraps like the bias corrected bootstrap percentile interval and the bootstrap percentile-t interval (see [Hansen](https://www.ssc.wisc.edu/~bhansen/econometrics/Econometrics.pdf) chapter 10 section 10.17 and 10.18). `vuong_tests6.py` simplifies this code.

Each of the different tests are designed to work with `statsmodels.api`. They take the likelihood `ll1` of each model. They also involve the gradient `grad1`, hessian `hess1`, parameters `params1` of each of the models. The following pseudo-code illustrates how to get the attributes from a generic likelihood model. The test requires a user defined `setup_test` function for getting these attributes from the models. This way the code will work with user defined models outside of `statsmodels`.

```python 
modeldel1 = your_model(y,X)
model1_fit = model1.fit()
ll1 = model1.loglikeobs(model1_fit.params)
grad1 =  model1.score_obs(model1_fit.params)
hess1 = model1.hessian(model1_fit.params)
params1 = model1_fit.params
```

## Main versions of the test

Each version of the test returns 1 when model 1 is selected 2 when model 2 is selected and 0 when no model is selected.

* `ndVuong` is the main function with the test designed by [Shi (2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE382).
* `regular_test` does the classical version of the test. It has the option to include the numerator bias correction with the `biascorrect` parameter. 
* `two_step_test` is designed to run a two-step classical version of the test.
* `bootstrap_test` is the main bootstrap version of the test

## Helper functions

The following helper functions are included:
* `compute_eigen2` is designed to compute the numerator correction described in the paper.
* `bootstrap_distr` computes all of the bootstrap test statistics with resampling.
* `choose_c` chooses the tuning parameter for the module.

## Visualizing performance

`vuong_plots.py` is a module intended for visualizing the performance of the test and evaluate different properties of the test.

# Main examples from the paper

The main examples in the paper are:
* `overlapping_reg` corresponds to the first set of Monte Carlo examples. `overlapping_reg` has its own readme explaining each of its' individual files.
* `shi_ex2` correspond to the tables in the second Monte Carlo analysis section. 
* `shi` is a port of the matlab code written for [Shi (2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE382) to python. I include the original matlab code for reference.

# Exploratory examples

* `multiple_models` explores how the test might generalize to multiple models.
* `splines` explores how the test might generalize to a non-parametric setting.
* `diff_norms` explores how the test might generalize the mean squared error as a selection criteria.

# Past examples no longer in the paper

* `auctions` contains examples similar to [Donald and Paarsch (1993)](https://www.jstor.org/stable/2526953?seq=1) involving testing functional form assumptions in auction models.
* `logit_probit` involves testing functional form assumptions in discrete choice models.
* `log_level` tests function form assumptions in the context of whether or not to take 'log' of the y variable in a regression model.
* `missing` tests Tobit vs OLS.
* subsampling explores an alternative subsampling bootstrap procedure similar to the one in (Romano and Shaikh (2012))[https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-6/On-the-uniform-asymptotic-validity-of-subsampling-and-the-bootstrap/10.1214/12-AOS1051.full].
* `nash` explores how the test might apply to different entry game models.



