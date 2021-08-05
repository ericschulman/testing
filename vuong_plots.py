import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm


def compute_eigen(yn,xn,setup_shi):
    ll1,grad1,hess1,params1,ll2, grad2,hess2,params2 = setup_shi(yn,xn)
    k1,k2 = params1.shape[0],params2.shape[0]
    hess1 = hess1/len(ll1)
    hess2 = hess2/len(ll2)
    k = k1 + k2
    
    #A_hat:
    A_hat1 = np.concatenate([hess1,np.zeros((k2,k1))])
    A_hat2 = np.concatenate([np.zeros((k1,k2)),-1*hess2])
    A_hat = np.concatenate([A_hat1,A_hat2],axis=1)

    #B_hat, covariance of the score...
    B_hat =  np.concatenate([grad1,-grad2],axis=1) #might be a mistake here..
    B_hat = np.cov(B_hat.transpose())


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


def compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2):
    
    n = ll1.shape[0]
    hess1 = hess1/n
    hess2 = hess2/n

    k1 = params1.shape[0]
    k2 = params2.shape[0]
    k = k1 + k2
    
    #A_hat:
    A_hat1 = np.concatenate([hess1,np.zeros((k2,k1))])
    A_hat2 = np.concatenate([np.zeros((k1,k2)),-1*hess2])
    A_hat = np.concatenate([A_hat1,A_hat2],axis=1)

    #B_hat, covariance of the score...
    B_hat =  np.concatenate([grad1,-grad2],axis=1) #might be a mistake here..
    B_hat = np.cov(B_hat.transpose())

    #compute eigenvalues for weighted chisq
    sqrt_B_hat= linalg.sqrtm(B_hat)
    W_hat = np.matmul(sqrt_B_hat,linalg.inv(A_hat))
    W_hat = np.matmul(W_hat,sqrt_B_hat)
    V,W = np.linalg.eig(W_hat)

    return V



def compute_analytic(yn,xn,setup_shi):
    nsims = 5000

    ll1,grad1,hess1,params1,ll2, grad2,hess2,params2 = setup_shi(yn,xn)
    hess1 = hess1/len(ll1)
    hess2 = hess2/len(ll2)
    k1,k2 = params1.shape[0],params2.shape[0]
    k = k1 + k2
    
    #A_hat:
    A_hat1 = np.concatenate([hess1,np.zeros((k2,k1))])
    A_hat2 = np.concatenate([np.zeros((k1,k2)),-1*hess2])
    A_hat = np.concatenate([A_hat1,A_hat2],axis=1)

    #B_hat, covariance of the score...
    B_hat =  np.concatenate([grad1,-grad2],axis=1) #might be a mistake here..
    B_hat = np.cov(B_hat.transpose())

    #B_hat, covariance of the score...
    B_hat =  np.concatenate([grad1,-grad2],axis=1) #might be a mistake here..
    B_hat = np.cov(B_hat.transpose())
    
    #compute eigenvalues for weighted chisq
    sqrt_B_hat= linalg.sqrtm(B_hat)
    W_hat = np.matmul(sqrt_B_hat,linalg.inv(A_hat))
    W_hat = np.matmul(W_hat,sqrt_B_hat)
    V,W = np.linalg.eig(W_hat)

    #simulate z_p as well...####
    abs_vecV = np.abs(V)-np.max(np.abs(V));
    rho_star = 1*(abs_vecV==0);
    rnorm = np.dot(rho_star.transpose(),rho_star)
    rho_star = np.dot( 1/np.sqrt(rnorm), rho_star)
    rho_star = np.array([rho_star])

    #simulate the normal distr asociated with parameters...
    np.random.seed()
    Z0 = np.random.normal( size=(nsims,k+1) )
    VZ1 = np.concatenate( [np.array([[1]]),rho_star.transpose() ])
    VZ2 = np.concatenate( [ rho_star,np.identity(k)])
    VZ = np.concatenate([VZ1,VZ2],axis=1)

    Z = np.matmul(Z0,linalg.sqrtm(VZ))
    Z_L = Z[:,0]            #$Z_Lambda$
    Z_p = Z[:,1:k+1]        #$Z_phi^\ast$
    
    #trace(V)  #diagonostic line
    tr_Vsq = (V*V).sum()
    V_nmlzd = V/np.sqrt(tr_Vsq) #V, normalized by sqrt(trVsq);

    J_Lmod = lambda sig,c: sig*Z_L - np.matmul(Z_p*Z_p,V_nmlzd)/2+ V_nmlzd.sum()/2
    
    J_omod = (lambda sig,c: sig**2 - 2*sig*np.matmul(Z_p,V_nmlzd*rho_star[0])
              + np.matmul(Z_p*Z_p,V_nmlzd*V_nmlzd) + c)
    
    final_distr = lambda sig,c: J_Lmod(sig,c)/np.sqrt(J_omod(sig,c))

    return final_distr(0,0), final_distr(10000,0)


def plot_true(gen_data,setup_shi):
    true_stats = []
    trials = 500
    for i in range(trials):
        np.random.seed()
        ys,xs,nobs = gen_data()
        ll1,grad1,hess1,k1,ll2,grad2,hess2,k2 = setup_shi(ys,xs)
        llr = (ll1 - ll2).sum()
        true_stats.append(2*llr)

    true_stats= np.array(true_stats)
    plt.hist( true_stats , density=True,bins=15, label="True",alpha=.60)
    return true_stats


def plot_analytic(yn,xn,nobs,setup_shi):
    n_sims = 5000
    model_eigs = compute_eigen(yn,xn,setup_shi)
    eigs_tile = np.tile(model_eigs,n_sims).reshape(n_sims,len(model_eigs))
    normal_draws = stats.norm.rvs(size=(n_sims,len(model_eigs)))
    weighted_chi = ((normal_draws**2)*eigs_tile).sum(axis=1)
    plt.hist(weighted_chi,density=True,bins=15,alpha=.60,label="Analytic")
    return weighted_chi


def plot_bootstrap(yn,xn,nobs,setup_shi):
    test_stats = []
    trials = 500
    for i in range(trials):
        subn = nobs
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),subn,replace=True)
        ys,xs = yn[sample],xn[sample]
        ll1,grad1,hess1,k1,ll2, grad2,hess2,k2 = setup_shi(ys,xs)
        llr = (ll1 - ll2).sum()
        test_stats.append(2*llr)
    
    plt.hist( test_stats, density=True,bins=15, label="Bootstrap",alpha=.60)
    return test_stats

############################  actual test stat ########################

def plot_true2(gen_data,setup_shi,trials=500):
    test_stats = []
    variance_stats  = []

    for i in range(trials):
        np.random.seed()
        ys,xs,nobs = gen_data()
        ll1,grad1,hess1,k1,ll2, grad2,hess2,k2 = setup_shi(ys,xs)
        
        ####messing around with recentering########
        
        V = compute_eigen2(ll1,grad1,hess1,k1,ll2, grad2,hess2,k2)
        ###################

        #llr = (ll1 - ll2).sum() +V_nmlzd.sum()/2
        llr = (ll1 - ll2).sum() +V.sum()/(2)
        omega2 = (ll1 - ll2).var() 
        test_stats.append(llr)
        variance_stats.append((np.sqrt(omega2*nobs)))
        
    test_stats = np.array(test_stats)
    test_stats = test_stats - test_stats.mean()
    varianc_stats = np.clip(variance_stats,.1,10000)
    result_stats = test_stats/variance_stats
    result_stats = result_stats[np.abs(result_stats) <= 8] #remove for plots..
    plt.hist(result_stats, density=True,bins=15, label="Bootstrap",alpha=.60)
    return result_stats


def plot_analytic2(yn,xn,nobs,setup_shi):
    overlap,normal =  compute_analytic(yn,xn,setup_shi)
    plt.hist(normal,density=True,bins=15,alpha=.60,label="Normal")
    plt.hist( overlap[np.abs(overlap) <= 8],density=True,bins=15,alpha=.60,label="Overlapping")
    return overlap[np.abs(overlap) <= 8],normal

###################################

def plot_bootstrap_pt(yn,xn,nobs,setup_shi,trials=1000,c=0):
    """final bootstrap that i ended up using 
    this one is fast... """
    ll1,grad1,hess1,params1,ll2,grad2,hess2,params2 = setup_shi(yn,xn)
    
    test_stats = []
    variance_stats = []
    llr = ll1-ll2
     
    for i in range(trials):
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),nobs,replace=True)
        llrs = llr[sample]
        test_stats.append( llrs.sum() )
        variance_stats.append( llrs.var() )

    #final product, bootstrap
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    test_stats = np.array(test_stats ) #
    test_stats = test_stats -test_stats.mean() #+ V.sum()/2  # recenter... makes figures easier
    variance_stats = np.sqrt(np.array(variance_stats)*nobs)

    plt.hist( test_stats/variance_stats, density=True,bins=15, label="Bootstrap",alpha=.60)
    return test_stats/variance_stats


def plot_bootstrap2(yn,xn,nobs,setup_shi,trials=500,c=0):
    """actually do the estimation on each iteration"""
    test_stats = []
    variance_stats  = []

    for i in range(trials):
        subn = nobs
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),subn,replace=True)
        ys,xs = yn[sample],xn[sample]
        ll1,grad1,hess1,k1,ll2, grad2,hess2,k2 = setup_shi(ys,xs)
        
        ####messing around with recentering########
        
        V = compute_eigen2(ll1,grad1,hess1,k1,ll2, grad2,hess2,k2)
        ###################

        #llr = (ll1 - ll2).sum() +V_nmlzd.sum()/2
        llr = (ll1 - ll2).sum() +V.sum()/(2)
        omega2 = (ll1 - ll2).var() 
        test_stats.append(llr)
        variance_stats.append((np.sqrt(omega2*nobs)))
        
    test_stats = np.array(test_stats)
    test_stats = test_stats - test_stats.mean()
    varianc_stats = np.clip(variance_stats,.1,10000)
    result_stats = test_stats/variance_stats
    result_stats = result_stats[np.abs(result_stats) <= 8] #remove for plots..
    plt.hist(result_stats, density=True,bins=15, label="Bootstrap",alpha=.60)
    return result_stats


##################################
################## ploting the k-stats
###################################

def plot_kstats_table(gen_data,setup_shi,figtitle=''):
    """make a table with the kstats"""

    #first make a figure...
    true_stats =  plot_true2(gen_data,setup_shi,trials=1000)

    yn,xn,nobs = gen_data()
    analytic_stats = plot_analytic2(yn,xn,nobs,setup_shi)
    #bootstrap_stats = plot_bootstrap_pt(yn,xn,nobs,setup_shi,trials=1000)
    bootstrap_stats = plot_bootstrap2(yn,xn,nobs,setup_shi,trials=1000)

    plt.legend()
    #then save it potentially
    if figtitle != '':
        print(figtitle)
        plt.savefig(figtitle)
    overlap,normal = analytic_stats
    plt.show()
    
        
    distr_list = [true_stats, bootstrap_stats,normal,overlap]
    distr_list_names = ['True','Bootstrap','Normal','Overlapping']
    n_kstats = 4
    
    moments = np.zeros((len(distr_list),n_kstats ))
    kstats =  np.zeros((len(distr_list),n_kstats ))

    #print moments/kstats
    for i in range(len(distr_list)):
        for j in range(n_kstats): #true_stats  
            m, k = stats.moment(distr_list[i], j+1), stats.kstat(distr_list[i], j+1)
            moments[i,j] = m
            kstats[i,j] = k

    print('\\begin{center}\n\\begin{tabular}{ccccccc}\n\\toprule')
    print('\\textbf{Test} & \\textbf{Mean} & \\textbf{Var} & \\textbf{K-Stat 3} & \\textbf{K-Stat 4} & \\textbf{Skew} & \\textbf{Kurtosis} \\\\ \\midrule' )
    for i in range(len(distr_list)): 
        print(distr_list_names[i]+' &',end='')
        print(' %.3f & %.3f & %.3f & %.3f &'%tuple(kstats[i]) ,end='')
        print(' %.3f & %.3f'%tuple(moments[i][2:4]) ,end='')
        print(' \\\\')
    print('\\bottomrule\n\\end{tabular}\n\\end{center}')
