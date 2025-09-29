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
        print('llr',llr,llr_split)
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
                            min_epsilon=1e-6, max_epsilon=10.0):
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
def bootstrap_distr(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
                    epsilon=0.5, trials=500, seed=None, biascorrect=False):
    nobs = ll1.shape[0]
    #print('---------')
    llr_reg, omega_reg, V, nobs = sw_test_stat(
        ll1, grad1, hess1, params1, ll2, grad2, hess2, params2, epsilon=epsilon,print_stuff=False
    )
    if biascorrect:
        llr_reg += V.sum() / 2
    test_stat = llr_reg / (omega_reg * np.sqrt(nobs))
    stat_dist = []

    test_stat0 = (ll1-ll2).sum()/np.sqrt( nobs*(ll1-ll2).var() )
    stats_s0 = []
    
    
    for i in range(trials):
        if seed is not None:
            np.random.seed(seed + i)
        sample = np.random.choice(np.arange(nobs), nobs, replace=True)
        ll1_s = ll1[sample]
        ll2_s = ll2[sample]

        # Don't need V for bootstrap stats (saves computation)
        llr_reg_s, omega_reg_s, nobs_in_s = sw_test_stat(
            ll1_s, grad1, hess1, params1, ll2_s, grad2, hess2, params2, epsilon=epsilon, compute_v=False,print_stuff=False
        )


        
        # bias correct V not directly available for bootstrap
        stat_s = (llr_reg_s - llr_reg) / (omega_reg_s * np.sqrt(nobs_in_s))
        stat_dist.append(stat_s)

        ###### real stat for comparison...
        llrs0 = ll1_s - ll2_s
        stats_s0.append(  llrs0.sum()/  np.sqrt(nobs*llrs0.var()) )
    if False:
        print('test_stat0', nobs_in_s, nobs, (np.array(stats_s0)-test_stat0).mean(),np.array(stats_s0).mean(),test_stat0 )
        print('test_stat', nobs_in_s, nobs, (np.array(stat_dist)-test_stat).mean(),np.array(stat_dist).mean(),test_stat )
    #print('----')
    return np.array(stat_dist), test_stat



def pairwise_bootstrap_distr(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
                    epsilon=0.5, trials=500, seed=None, biascorrect=False):
    rng = np.random.default_rng(seed)
    nobs = ll1.shape[0]

    # Compute observed statistic
    llr_reg, omega_reg, V, _ = sw_test_stat(
        ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
        epsilon=epsilon, print_stuff=False
    )
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
        stat_dist.append(stat_s)
    
    return np.array(stat_dist) , test_stat


def sw_bs_test_helper(stat_dist, stat_obs, alpha=.05, left=True, right=True, print_stuff=False):
    cv_upper = np.percentile(stat_dist, 100 * (1 - alpha / 4)) #need to make the alpha a little lower... i think im not doing enough draws...
    cv_lower = np.percentile(stat_dist, 100 * (alpha / 4))
    if print_stuff:
        print(f"mean={stat_dist.mean():.3f}, cv_lower={cv_lower:.3f}, stat_obs={stat_obs:.3f}, cv_upper={cv_upper:.3f}")
    out = 0
    if right and (stat_obs > cv_upper):
        out = 1
    if left and (stat_obs < cv_lower):
        out += 2
    return out

def sw_bs_test(ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
               alpha=.05, trials=500, epsilon=0.5, biascorrect=False, seed=None, print_stuff=False,pairwise=False):
    
    stat_dist, test_stat = None,None
    if not pairwise:
        stat_dist, test_stat = bootstrap_distr(
            ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
            epsilon=epsilon, trials=trials, seed=seed, biascorrect=biascorrect
        )
    else:
        stat_dist, test_stat = pairwise_bootstrap_distr(
            ll1, grad1, hess1, params1, ll2, grad2, hess2, params2,
            epsilon=epsilon, trials=trials, seed=seed, biascorrect=biascorrect
        )
    return sw_bs_test_helper(stat_dist, test_stat, alpha=alpha, print_stuff=print_stuff)




def monte_carlo(total,gen_data,setup_shi,skip_boot=False,skip_shi=False,trials=500,
    biascorrect=False,alpha=.05,adapt_c=True,print_stuff=True,epsilon=.5,data_tuned_epsilon=False):
    reg = np.array([0, 0 ,0])
    twostep = np.array([0, 0 ,0])

    sw = np.array([0, 0 ,0])
    shi = np.array([0, 0 ,0])

    boot1 = np.array([0, 0 ,0])
    boot2 = np.array([0, 0 ,0])

    sw_test_opt = np.array([0, 0 ,0])
    boot3 = np.array([0, 0 ,0])

    omega = 0
    llr = 0
    var = 0
    for i in range(total):
        np.random.seed()
        yn,xn,nobs = gen_data()
        
        #update the llr
        ll1,grad1,hess1,params1,ll2,grad2,hess2,params2 = setup_shi(yn,xn)
        llrn = (ll1 - ll2).sum()
        omegan = np.sqrt( (ll1 -ll2).var())
        llr = llr +llrn
        var = llrn**2 + var
        omega = omega +omegan


        #run the test
        reg_index = regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,biascorrect=False,alpha=alpha)
        twostep_index = two_step_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,biascorrect=False,alpha=alpha)
        sw_index = sw_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,epsilon=epsilon,alpha=alpha,biascorrect=biascorrect #TODO messing around... fix this back
            ,print_stuff=False)
        shi_index,boot_index1,boot_index2,boot_index3 = 0,0,0,0 #take worst case for now...

        shi_index=0
        if not skip_shi:
            shi_index = ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,alpha=alpha,adapt_c=adapt_c)

        #bootstrap indexes....
        boot_index1,boot_index2 = 0,0
        if not skip_boot:

            print_stuff_i =  ((i%25) == 0 ) and print_stuff
            
            # boot1: standard bootstrap
            boot_index1 = sw_bs_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,
                                    alpha=alpha,trials=trials,epsilon=epsilon,biascorrect=biascorrect,
                                    seed=None,print_stuff=False,pairwise=False)
            # boot2: pairwise bootstrap
            boot_index2 = sw_bs_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,
                                    alpha=alpha,trials=trials,epsilon=epsilon,biascorrect=biascorrect,
                                    seed=None,print_stuff=False,pairwise=True)

        
        sw_opt_index,boot_index3 = 0,0
        if data_tuned_epsilon:
            epsilon_opt = compute_optimal_epsilon(ll1, grad1, hess1, params1,
                 ll2, grad2, hess2, params2, alpha=alpha)
            sw_opt_index = sw_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,epsilon=epsilon_opt,
                alpha=alpha,biascorrect=biascorrect,print_stuff=False)
            # boot3: pairwise bootstrap w/ optimal epsilon
            boot_index3 = sw_bs_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,
                                    alpha=alpha,trials=trials,epsilon=epsilon_opt,biascorrect=biascorrect,
                                    seed=None,print_stuff=False,pairwise=True)

        reg[reg_index] = reg[reg_index] + 1
        twostep[twostep_index] = twostep[twostep_index] +1
        
        sw[sw_index] = sw[sw_index] +1
        shi[shi_index] = shi[shi_index] + 1


        boot1[boot_index1] = boot1[boot_index1] + 1
        boot2[boot_index2] = boot2[boot_index2] + 1

        sw_test_opt[sw_opt_index] = sw_test_opt[sw_opt_index] +1
        boot3[boot_index3] = boot3[boot_index3] + 1
        
    return reg/total,twostep/total,sw/total,boot1/total,boot2/total,sw_test_opt/total,boot3/total,shi/total,llr/total,np.sqrt( (var/total-(llr/total)**2) ),omega*np.sqrt(nobs)/total
    

def print_mc(mc_out,data_tuned_epsilon=False):
    reg,twostep, sw, boot1,boot2,sw_test_opt,boot3,shi, llr,std, omega =  mc_out
    print('\\begin{tabular}{|c|c|c|c|c|c|c|}')
    print('\\hline')
    print('Model &  Normal & Two-Step & SW Test & Naive Bootstrap & Pairwise Bootstrap & Shi (2015) \\\\ \\hline \\hline')
    labels = ['No selection', 'Model 1', 'Model 2']
    for i in range(3): 
        if data_tuned_epsilon:
            print('%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f   \\\\'%(labels[i], round(reg[i],2),round(twostep[i],2),round(sw_test_opt[i],2),round(boot1[i],2),round(boot3[i],2),round(shi[i],2)))
        else:
            print('%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f   \\\\'%(labels[i], round(reg[i],2),round(twostep[i],2),round(sw[i],2),round(boot1[i],2),round(boot2[i],2),round(shi[i],2)))
    print('\\hline')
    print('\\end{tabular}')


def print_mc2(mc_out):
    reg,twostep, sw, boot1,boot2,sw_test_opt,boot3,shi, llr,std, omega =  mc_out
    print('\\begin{tabular}{|c|c|c|}')
    print('\\hline')
    print('Model &  SW Test & Pairwise Bootstrap  \\\\ \\hline \\hline')
    labels = ['No selection', 'Model 1', 'Model 2']
    for i in range(3): 
        print('%s & %.2f & %.2f \\\\'%(labels[i], round(sw_test_opt[i],2),round(boot3[i],2) ))
    print('\\hline')
    print('\\end{tabular}')







