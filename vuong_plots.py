import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm


def compute_eigen(yn,xn,setup_shi):
    ll1,grad1,hess1,ll2,k1, grad2,hess2,k2 = setup_shi(yn,xn)
    hess1 = hess1/len(ll1)
    hess2 = hess2/len(ll2)
    
    k = k1 + k2
    n = len(ll1)
    
    #A_hat:
    A_hat1 = np.concatenate([hess1,np.zeros((k1,k2))])
    A_hat2 = np.concatenate([np.zeros((k2,k1)),-1*hess2])
    A_hat = np.concatenate([A_hat1,A_hat2],axis=1)

    #B_hat, covariance of the score...
    B_hat =  np.concatenate([grad1,-grad2],axis=1) #might be a mistake here..
    B_hat = np.cov(B_hat.transpose())
    #print(B_hat[0:3,3:])
    
    #compute eigenvalues for weighted chisq
    sqrt_B_hat= linalg.sqrtm(B_hat)
    W_hat = np.matmul(sqrt_B_hat,linalg.inv(A_hat))
    W_hat = np.matmul(W_hat,sqrt_B_hat)
    V,W = np.linalg.eig(W_hat)
    return V

def plot_true(gen_data,setup_shi):
    true_stats = []
    trials = 200
    for i in range(trials):
        np.random.seed()
        ys,xs,nobs = gen_data()
        ll1,grad1,hess1,ll2,k1, grad2,hess2,k2 = setup_shi(ys,xs)
        llr = (ll1 - ll2).sum()
        true_stats.append(2*llr)

    plt.hist( true_stats, density=True,bins=15, label="True",alpha=.75)
    return true_stats

def plot_analytic(yn,xn,nobs,setup_shi):
    n_sims = 5000
    model_eigs = compute_eigen(yn,xn,setup_shi)
    eigs_tile = np.tile(model_eigs,n_sims).reshape(n_sims,len(model_eigs))
    normal_draws = stats.norm.rvs(size=(n_sims,len(model_eigs)))
    weighted_chi = ((normal_draws**2)*eigs_tile).sum(axis=1)
    plt.hist(weighted_chi,density=True,bins=20,alpha=.75,label="Analytic")
    return weighted_chi

def plot_bootstrap(yn,xn,nobs,setup_shi):
    test_stats = []
    trials = 200
    for i in range(trials):
        subn = nobs
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),subn,replace=True)
        ys,xs = yn[sample],xn[sample]
        ll1,grad1,hess1,ll2,k1, grad2,hess2,k2 = setup_shi(ys,xs)
        llr = (ll1 - ll2).sum()
        test_stats.append(2*llr)
    
    plt.hist( test_stats, density=True,bins=15, label="Bootstrap",alpha=.75)
    return test_stats