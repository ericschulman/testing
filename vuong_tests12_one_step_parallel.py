import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm
from vuong_test_base import *

#########################################################################################

def sw_test_stat(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2, epsilon=.5,compute_v=True,print_stuff=False):
    nobs = ll1.shape[0]

    idx_even = np.arange(0, nobs, 2)         # 0-based indices: 0,2,4,...
    idx_odd  = np.arange(1, nobs, 2)         # 1,3,5,...

    # Regular loglikelihood ratio statistic
    llr = (ll1 - ll2).sum()
    
    # Split-sample difference statistic
    llr_split = ll1[idx_even].sum() - ll2[idx_odd].sum()

    # Regularized numerator (as in your expression)
    llr_reg = llr + epsilon * llr_split

    # Main variance
    omega2 = (ll1 - ll2).var()
    
    # Split-group variances
    omega_A2 = ll1[idx_even].var()
    omega_B2 = ll2[idx_odd].var()

    # Regularized variance
    omega_reg2 = (1 + epsilon) * omega2 + (epsilon**2 / 2) * (omega_A2 + omega_B2)
    omega_reg = np.sqrt(omega_reg2)

    if print_stuff:
        print('llr',llr,llr_split,epsilon)
        print('omega',omega2,omega_A2,omega_B2,np.sqrt(omega2),np.sqrt(omega_reg2))
    if not compute_v:
        return llr_reg,omega_reg,nobs 
        
    V = compute_eigen2(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2) # If you want to try to bias-correct you can, this is optional
    return llr_reg,omega_reg,V,nobs 
        
    


def sw_test(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2, epsilon=.5,
            alpha=.05,  biascorrect=False,print_stuff=False):
    
    llr_reg,omega_reg,V,nobs =  sw_test_stat(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2, epsilon=epsilon)

    if biascorrect:
        llr_reg += V.sum() / 2

    test_stat = llr_reg / (omega_reg * np.sqrt(nobs))
    if print_stuff:
        print(epsilon,test_stat)
    # Two-sided test
    reject_high = (test_stat >= norm.ppf(1 - alpha / 2))
    reject_low  = (test_stat <= norm.ppf(alpha / 2))
    return int(reject_high) + 2 * int(reject_low)



##########################################################################

def _estimate_HV(grad, hess_total, ridge=1e-8):
    """
    grad: (n,p) per-observation score of the log-likelihood
    hess_total: (p,p) full-sample Hessian of the log-likelihood (sum over i)
    Returns:
    H_hat = -E[∂² log f]  (per-observation average curvature)
    V_hat = E[s s′]       (uncentered score second moment)
    """
    n = grad.shape[0]
    H_hat = -hess_total / n
    S = grad - grad.mean() # i think its supposed to be centered....
    V_hat = (S.T @ S) / n
    return H_hat, V_hat

def _trace_HinvV(H, V):
    return float(np.trace(np.linalg.solve(H, V)))


def compute_optimal_epsilon(ll1, grad1, hess1, params1,
                 ll2, grad2, hess2, params2, alpha=0.05, ridge=1e-8,
                            min_epsilon=1e-6, max_epsilon=0.5):
    nobs = ll1.shape[0]
    idx_even = np.arange(0, nobs, 2)
    idx_odd  = np.arange(1, nobs, 2)

    # Variances
    sigma2_hat  = max(float(np.var(ll1 - ll2, ddof=0)),.5) #NOTE cheating a little bit with epsilon... keep it big enough
    sigmaA2_hat = float(np.var(ll1[idx_even], ddof=0)) if idx_even.size > 1 else 0.0
    sigmaB2_hat = float(np.var(ll2[idx_odd],  ddof=0)) if idx_odd.size  > 1 else 0.0
    sigma_hat   = np.sqrt(max(sigma2_hat, 1e-12))

    # Estimate (H, V) for each model
    H1_hat, V1_hat = _estimate_HV(grad1, hess1, ridge=ridge)
    H2_hat, V2_hat = _estimate_HV(grad2, hess2, ridge=ridge)
    try:
        tr1 = abs(_trace_HinvV(H1_hat, V1_hat))
        tr2 = abs(_trace_HinvV(H2_hat, V2_hat))
        tr_max = max(tr1, tr2)
    except np.linalg.LinAlgError:
        tr_max = 1.0

    # Paper constants
    z = norm.ppf(1 - alpha/2.0)
    phi_z = norm.pdf(z)
    delta_star_hat = (sigma_hat / 2.0) * (z - np.sqrt(4.0 + z**2))
    phi_arg = z - (delta_star_hat / max(sigma_hat, 1e-12))
    phi_term = norm.pdf(phi_arg)

    denom_PL = 4.0 * (sigma_hat**3 + 1e-12)
    CPL_num = delta_star_hat * (sigma_hat**2 - 2.0 * (sigmaA2_hat + sigmaB2_hat))
    C_PL_hat = phi_term * CPL_num / denom_PL

    denom_SD = np.sqrt(max((sigmaA2_hat + sigmaB2_hat)/2.0, 1e-12))
    C_SD_hat = 2.0 * phi_z * tr_max / denom_SD

    lnln_term = max(np.log(np.log(max(nobs, 3))), 1e-6)
    fallback = nobs**(-1/6) * (lnln_term**(1/3))

    valid = (np.isfinite(C_PL_hat) and np.isfinite(C_SD_hat) and
             C_PL_hat > 0 and C_SD_hat > 0)

    if valid:
        eps_hat = ((C_SD_hat/C_PL_hat)**(1/3)) * (nobs**(-1/6)) * (lnln_term**(1/3))
    else:
        eps_hat = fallback

    return float(np.clip(eps_hat, min_epsilon, max_epsilon))



############################################################################
def normal_one_step_factory(args):
    """One-step quadratic updates using the supplied total Hessians.

    Retain the name for existing notebooks. Use the average log-likelihood
    Hessian for each observation: exact for constant-Hessian quadratic
    likelihoods, an approximation otherwise. This alone does not establish
    bootstrap refinement.
    """
    n = len(args[0])

    def prepare(ll, grad, hess):
        scores = np.asarray(grad).reshape(n, -1)
        H = np.asarray(hess) / n
        # Solve once per original dataset, outside bootstrap loops.
        directions = -np.linalg.solve(H, scores.T).T
        return np.asarray(ll), scores, directions, H

    a, sf, df, Hf = prepare(args[0], args[1], args[2])
    b, sg, dg, Hg = prepare(args[4], args[5], args[6])

    def update(sample):
        delta_f = df[sample].mean(axis=0)
        delta_g = dg[sample].mean(axis=0)
        a_s = a[sample] + sf[sample] @ delta_f + 0.5 * (delta_f @ Hf @ delta_f)
        b_s = b[sample] + sg[sample] @ delta_g + 0.5 * (delta_g @ Hg @ delta_g)
        return a_s, b_s

    return update



def bootstrap_distr(
    ll1, grad1, hess1, params1,
    ll2, grad2, hess2, params2,
    epsilon=0.5, trials=500, seed=None, biascorrect=False,
    one_step_update=None
):
    nobs = ll1.shape[0]

    if nobs % 2 != 0:
        raise ValueError("This implementation requires an even sample size.")

    # ---------------------------------------------------------------
    # Observed statistic: unchanged from your original implementation.
    # ---------------------------------------------------------------
    llr_reg, omega_reg, V, _ = sw_test_stat(
        ll1, grad1, hess1, params1,
        ll2, grad2, hess2, params2,
        epsilon=epsilon,
        print_stuff=False
    )

    observed_numerator = llr_reg

    # Original-sample correction: no eigenvalue calculations inside the loop.
    bias_correction = V.sum() / 2.0 if biascorrect else 0.0
    observed_numerator += bias_correction

    test_stat = observed_numerator / (omega_reg * np.sqrt(nobs))


    D = (ll1 - ll2).sum()
    bootstrap_center = (1.0 + epsilon / 2.0) * D

    stat_dist = np.empty(trials)

    for i in range(trials):
        # Retain your original random-number behavior.
        if seed is not None:
            np.random.seed(seed + i)

        sample = np.random.choice(
            np.arange(nobs), size=nobs, replace=True
        )

        if one_step_update is None:
            ll1_s, ll2_s = ll1[sample], ll2[sample]
        else:
            ll1_s, ll2_s = one_step_update(sample)

        # Recompute the denominator from these likelihoods; epsilon stays fixed.
        llr_reg_s, omega_reg_s, nobs_in_s = sw_test_stat(
            ll1_s, grad1, hess1, params1,
            ll2_s, grad2, hess2, params2,
            epsilon=epsilon,
            compute_v=False,
            print_stuff=False
        )

        stat_dist[i] = (
            (llr_reg_s - bootstrap_center)
            / (omega_reg_s * np.sqrt(nobs_in_s))
        )

    return stat_dist, test_stat


def pairwise_bootstrap_distr(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
                    epsilon=0.5, trials=500, seed=None, biascorrect=False
                    ,print_stuff=False ):
    rng = np.random.default_rng(seed)
    nobs = ll1.shape[0]

    # Compute observed statistic
    llr_reg, omega_reg, V, _ = sw_test_stat(
        ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
        epsilon=epsilon, print_stuff=print_stuff
    )
    
    if print_stuff:
        print('llr_reg,omega_reg',llr_reg, omega_reg, V.sum() / 2)

    if biascorrect:
        llr_reg += V.sum() / 2
    test_stat = llr_reg / (omega_reg * np.sqrt(nobs))

    # Indices for even/odd destinations and source pairs
    idx_even = np.arange(0, nobs, 2)
    idx_odd  = idx_even + 1
    m = idx_even.size
    pair_ids = np.arange(m)  # pair j corresponds to source indices (2j, 2j+1)
    
    stat_dist = []
    for b in range(trials):
        draw = rng.choice(pair_ids, size=m, replace=True)
    
        # Source indices for drawn pairs
        src_even = idx_even[draw]
        src_odd  = idx_odd[draw]
    
        # Build bootstrap arrays by placing drawn pairs into their parity slots
        ll1_s = np.empty_like(ll1)
        ll2_s = np.empty_like(ll2)
        ll1_s[idx_even] = ll1[src_even]
        ll2_s[idx_even] = ll2[src_even]
        ll1_s[idx_odd]  = ll1[src_odd]
        ll2_s[idx_odd]  = ll2[src_odd]
    
        # Compute bootstrap statistic (no need to compute V each time)
        llr_reg_s, omega_reg_s, nobs_in_s = sw_test_stat(
            ll1_s, grad1, hess1, params1, ll2_s, grad2, hess2, params2,
            epsilon=epsilon, compute_v=False, print_stuff=False
        )
        stat_s = (llr_reg_s-llr_reg) / (omega_reg_s * np.sqrt(nobs_in_s))
        if biascorrect:
            stat_s = (llr_reg_s-llr_reg+V.sum()/(2)) / (omega_reg_s * np.sqrt(nobs_in_s))
      
        stat_dist.append(stat_s)
    stat_dist = np.array(stat_dist)
    return stat_dist , test_stat


def sw_bs_test(
    ll1, grad1, hess1, params1,
    ll2, grad2, hess2, params2,
    alpha=.05, trials=500, epsilon=.5,
    biascorrect=False, seed=None, print_stuff=False
):
    stat_dist, test_stat = bootstrap_distr(
        ll1, grad1, hess1, params1,
        ll2, grad2, hess2, params2,
        epsilon=epsilon,
        trials=trials,
        seed=seed,
        biascorrect=biascorrect
    )

    out = sw_bs_test_helper(
        stat_dist, test_stat,
        alpha=alpha,
        print_stuff=print_stuff
    )

    if print_stuff:
        print("returned out =", out)

    return out


def sw_bs_test_helper(stat_dist, stat_obs, alpha=.05, left=True, right=True, print_stuff=False):
    cv_upper = np.percentile(stat_dist, 100 * (1 - alpha / 2)) #need to make the alpha a little lower... i think im not doing enough draws...
    cv_lower = np.percentile(stat_dist, 100 * (alpha / 2))
    if print_stuff:
        print(f"mean={stat_dist.mean():.3f}, cv_lower={cv_lower:.3f}, stat_obs={stat_obs:.3f}, cv_upper={cv_upper:.3f}")
    out = 0
    if right and (stat_obs > cv_upper):
        out = 1
    if left and (stat_obs < cv_lower):
        out = 2
    return out

def sw_bs_test(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
               alpha=.05, trials=500, epsilon=0.5, biascorrect=False, seed=None, print_stuff=False,pairwise=False,
               one_step_update=None):
    
    if pairwise and one_step_update is not None:
        raise ValueError("One-step updates are only supported for singleton draws")
    stat_dist, test_stat = None,None
    if not pairwise:
        stat_dist, test_stat = bootstrap_distr(
            ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
            epsilon=epsilon, trials=trials, seed=seed, biascorrect=biascorrect,
            one_step_update=one_step_update
        )
    else:
        stat_dist, test_stat = pairwise_bootstrap_distr(
            ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
            epsilon=epsilon, trials=trials, seed=seed, biascorrect=biascorrect, print_stuff=print_stuff
        )

    out = sw_bs_test_helper(stat_dist, test_stat, alpha=alpha, print_stuff=print_stuff)
    if print_stuff:
        print("returned out =", out)

    return out



def _monte_carlo_serial(
    total, gen_data, setup_shi,
    skip_boot=False, skip_shi=False, trials=500,
    biascorrect=False, alpha=.05, adapt_c=True,
    print_stuff=True, epsilon=.5, data_tuned_epsilon=False,
    one_step_factory=None
):
    reg = np.zeros(3, dtype=int)
    twostep = np.zeros(3, dtype=int)
    sw = np.zeros(3, dtype=int)
    shi = np.zeros(3, dtype=int)

    boot1 = np.zeros(3, dtype=int)        # Fixed singleton
    sw_test_opt = np.zeros(3, dtype=int)
    boot3 = np.zeros(3, dtype=int)        # Tuned singleton

    omega = 0.0
    llr = 0.0
    var = 0.0

    for i in range(total):
        yn, xn, nobs = gen_data()

        args = setup_shi(yn, xn)
        # Prepare once per dataset; reuse for fixed AND tuned epsilon.
        update = (one_step_factory(args)
                  if one_step_factory is not None and not skip_boot else None)
        ll1, grad1, hess1, params1, ll2, grad2, hess2, params2 = args

        llrn = (ll1 - ll2).sum()
        omegan = np.sqrt((ll1 - ll2).var())

        llr += llrn
        var += llrn**2
        omega += omegan

        # Original non-bootstrap procedures, unchanged.
        reg_index = regular_test(
            *args, biascorrect=False,
            alpha=alpha, print_stuff=False
        )
        twostep_index = two_step_test(
            *args, biascorrect=False, alpha=alpha
        )
        sw_index = sw_test(
            *args, epsilon=epsilon, alpha=alpha,
            biascorrect=False, print_stuff=False
        )

        reg[reg_index] += 1
        twostep[twostep_index] += 1
        sw[sw_index] += 1

        if not skip_shi:
            shi_index = ndVuong(
                *args, alpha=alpha, adapt_c=adapt_c
            )
            shi[shi_index] += 1

        if not skip_boot:
            # Correctly centered singleton, fixed epsilon.
            boot_index1 = sw_bs_test(
                *args, alpha=alpha, trials=trials,
                epsilon=epsilon,
                biascorrect=biascorrect,
                seed=None, print_stuff=False, one_step_update=update
            )
            boot1[boot_index1] += 1

        if data_tuned_epsilon:
            epsilon_opt = compute_optimal_epsilon(
                *args, alpha=alpha
            )

            sw_opt_index = sw_test(
                *args, epsilon=epsilon_opt,
                alpha=alpha, biascorrect=False,
                print_stuff=False
            )
            sw_test_opt[sw_opt_index] += 1

            if not skip_boot:
                # Correctly centered singleton, tuned epsilon.
                boot_index3 = sw_bs_test(
                    *args, alpha=alpha, trials=trials,
                    epsilon=epsilon_opt,
                    biascorrect=biascorrect,
                    seed=None, print_stuff=False, one_step_update=update
                )
                boot3[boot_index3] += 1

        if print_stuff and (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{total}")

    missing = np.full(3, np.nan)
    std = np.sqrt(max(var / total - (llr / total)**2, 0.0))

    # Same tuple layout as before.
    return (
        reg / total,
        twostep / total,
        sw / total,
        boot1 / total if not skip_boot else missing.copy(),
        missing.copy(),  # Former boot2 position: pairwise removed
        sw_test_opt / total if data_tuned_epsilon else missing.copy(),
        boot3 / total
        if data_tuned_epsilon and not skip_boot else missing.copy(),
        shi / total if not skip_shi else missing.copy(),
        llr / total,
        std,
        omega * np.sqrt(nobs) / total
    )
    

def _monte_carlo_one_replication(seed, gen_data, setup_shi, options):
    """One independent simulated dataset; top-level for process workers."""
    np.random.seed(int(seed))
    sample_size = []

    def recorded_gen_data():
        y, x, n = gen_data()
        sample_size.append(n)
        return y, x, n

    result = _monte_carlo_serial(
        1, recorded_gen_data, setup_shi, print_stuff=False, **options
    )
    return result, sample_size[0]


def _combine_monte_carlo_replications(replications):
    """Pool counts and moments, rather than averaging within-run SDs."""
    results = [r[0] for r in replications]
    sizes = np.asarray([r[1] for r in replications])
    probabilities = tuple(
        np.mean(np.stack([r[j] for r in results]), axis=0)
        for j in range(8)
    )
    likelihood_sums = np.asarray([r[8] for r in results])
    mean_llr = likelihood_sums.mean()
    std_llr = np.sqrt(max(np.mean(likelihood_sums ** 2) - mean_llr ** 2, 0.0))
    # Match the original diagnostic, including its use of the last sample size.
    mean_omega = np.mean([r[10] / np.sqrt(n) for r, n in zip(results, sizes)])
    return probabilities + (mean_llr, std_llr, mean_omega * np.sqrt(sizes[-1]))


def monte_carlo(
    total, gen_data, setup_shi,
    skip_boot=False, skip_shi=False, trials=500,
    biascorrect=False, alpha=.05, adapt_c=True,
    print_stuff=True, epsilon=.5, data_tuned_epsilon=False,
    one_step_factory=None, n_jobs=1, seed=None
):
    """Original one-step Monte Carlo with optional process parallelism.

    n_jobs=1, seed=None preserves the original sequential path exactly.
    n_jobs=2 uses two processes; n_jobs=-1 uses available CPUs (up to total).
    joblib/loky serializes notebook functions and closures, including on Windows.
    Each task is one dataset, with its bootstrap draws kept sequential.

    seed initializes independent per-dataset streams in the dispatched path.
    The supplied base routines call np.random.seed() with no argument, so seed
    does not guarantee fully reproducible results for those routines or later
    draws. The base routines themselves are left unchanged.
    """
    if not isinstance(total, (int, np.integer)) or total < 1:
        raise ValueError("total must be a positive integer")
    if not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")
    options = dict(
        skip_boot=skip_boot, skip_shi=skip_shi, trials=trials,
        biascorrect=biascorrect, alpha=alpha, adapt_c=adapt_c,
        epsilon=epsilon, data_tuned_epsilon=data_tuned_epsilon,
        one_step_factory=one_step_factory
    )
    if n_jobs == 1 and seed is None:
        return _monte_carlo_serial(
            total, gen_data, setup_shi, print_stuff=print_stuff, **options
        )

    # SeedSequence avoids inheriting the same NumPy stream in every worker.
    seeds = [child.generate_state(1)[0]
             for child in np.random.SeedSequence(seed).spawn(total)]
    if n_jobs == 1:
        replications = [
            _monte_carlo_one_replication(s, gen_data, setup_shi, options)
            for s in seeds
        ]
    else:
        try:
            from joblib import Parallel, delayed, effective_n_jobs, parallel_backend
        except ImportError as exc:
            raise ImportError("Parallel runs require joblib: pip install joblib") from exc
        workers = min(total, effective_n_jobs(n_jobs))
        # Prevent each process from starting its own large BLAS thread pool.
        with parallel_backend("loky", inner_max_num_threads=1):
            replications = Parallel(
                n_jobs=workers, batch_size="auto", pre_dispatch="2*n_jobs",
                verbose=5 if print_stuff else 0
            )(
                delayed(_monte_carlo_one_replication)(
                    s, gen_data, setup_shi, options
                ) for s in seeds
            )
    return _combine_monte_carlo_replications(replications)


def print_mc(mc_out, data_tuned_epsilon=False):
    reg, twostep, sw, boot1, _, sw_opt, boot_opt, shi, *_ = mc_out

    headers = ["Normal", "Two-Step", "SW Test", "Singleton", "Shi (2015)"]
    arrays = [reg, twostep, sw, boot1, shi]

    if data_tuned_epsilon:
        headers += ["Tuned SW", "Tuned Singleton"]
        arrays += [sw_opt, boot_opt]

    print(r"\begin{tabular}{|" + "c|" * (len(headers) + 1) + "}")
    print(r"\hline")
    print("Model & " + " & ".join(headers) + r" \\ \hline")

    for i, label in enumerate(["No selection", "Model 1", "Model 2"]):
        values = [f"{arr[i]:.3f}" for arr in arrays]
        print(label + " & " + " & ".join(values) + r" \\")

    print(r"\hline")
    print(r"\end{tabular}")


def print_mc2(mc_out):
    _, _, _, _, _, sw_opt, boot_opt, _, *_ = mc_out

    print(r"\begin{tabular}{|c|c|c|}")
    print(r"\hline")
    print(r"Model & Tuned SW & Tuned Singleton \\ \hline")

    for i, label in enumerate(["No selection", "Model 1", "Model 2"]):
        print(f"{label} & {sw_opt[i]:.3f} & {boot_opt[i]:.3f}" + r" \\")

    print(r"\hline")
    print(r"\end{tabular}")