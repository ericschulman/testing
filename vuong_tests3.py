import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm


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


def regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,trials=500,c=0):
    nobs = ll1.shape[0]
    
    llr = (ll1 - ll2).sum()
    omega = np.sqrt( (ll1 -ll2).var())
    test_stat = llr/(omega*np.sqrt(nobs))
    return 1*(test_stat >= 1.96) + 2*( test_stat <= -1.96)


def bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,trials=500,c=0):
    nobs = ll1.shape[0]
    
    test_stats = []
    variance_stats = []
    llr = ll1-ll2
     
    for i in range(trials):
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),nobs,replace=True)
        llrs = llr[sample]
        test_stats.append( llrs.sum() )
        variance_stats.append( llrs.var() )
    
    #final product
    V = compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    test_stats = np.array(test_stats)+ V.sum()/2
    variance_stats = np.sqrt(variance_stats)*np.sqrt(nobs) + c*(V*V).sum()/(nobs)

    return test_stats,variance_stats


def bootstrap_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=0,trials=500):
    nobs = ll1.shape[0]

    #set up bootstrap distr
    test_stats,variance_stats = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=c,trials=trials)
    test_stats = test_stats/variance_stats
    
    #set up confidence intervals
    cv_lower = np.percentile(test_stats, 2.5, axis=0)
    cv_upper = np.percentile(test_stats, 97.5, axis=0) 

    return  2*(0 >= cv_upper) + 1*(0 <= cv_lower)


def bootstrap_bc(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=0,trials=500):
    #setup test stat
    nobs = ll1.shape[0]
    
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    omega2 = (ll1 - ll2).var() + c*(V*V).sum()/(nobs)
    llr = (ll1 - ll2).sum() +V.sum()/(2)
    test_stat = llr/np.sqrt(omega2*nobs)
    
    #set up bootstrap distr
    test_stats,variance_stats = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=c,trials=trials)
    test_stats = test_stats/variance_stats

    #estimate median "bias"
    z_0 = norm.ppf( (test_stats<=test_stat).mean() ) #measure of median bias
    z_alpha = norm.ppf(np.linspace(.0001, .9999, 2*test_stats.shape[0]))
    x_alpha = norm.cdf(z_alpha + 2*z_0)
   
    #adjust quantiles accordingly
    x_lower = np.percentile(x_alpha, 2.5, axis=0)
    x_upper = np.percentile(x_alpha, 97.5, axis=0)

    #set up confidence intervals
    cv_lower = np.percentile(test_stats, x_lower*100, axis=0)
    cv_upper = np.percentile(test_stats, x_upper*100, axis=0)

    return  2*(0 >= cv_upper) + 1*(0 <= cv_lower)


def bootstrap_test_pivot(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=0,trials=500):
    #setup test stat
    ll1,grad1,hess1,params1,ll2,grad2,hess2,params2
    nobs = ll1.shape[0]
    
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    omega2 = (ll1 - ll2).var() + c*(V*V).sum()/(nobs)
    llr = (ll1 - ll2).sum() +V.sum()/(2)
    test_stat = llr/np.sqrt(omega2*nobs)
    
    #set up bootstrap distr
    test_stats,variance_stats = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=c,trials=trials)
    test_stats = test_stats/variance_stats
    
    #set up confidence intervals
    cv_lower = 2*test_stat - np.percentile(test_stats, 97.5, axis=0)
    cv_upper = 2*test_stat -  np.percentile(test_stats, 2.5, axis=0)

    return  2*(0 >= cv_upper) + 1*(0 <= cv_lower)


def bootstrap_test_pt(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=0,trials=500):
    #setup test stat
    nobs = ll1.shape[0]
    
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    omega2 = (ll1 - ll2).var() + c*(V*V).sum()/(nobs)
    llr = (ll1 - ll2).sum() +V.sum()/(2)
    test_stat = llr/np.sqrt(omega2*nobs)
    
    #set up bootstrap distr
    test_stats,variance_stats = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=c,trials=trials)
    
    #set up confidence intervals
    test_stats = (test_stats- llr)/np.sqrt(omega2*nobs)
    cv_lower = llr - np.percentile(test_stats, 97.5, axis=0)*np.sqrt(omega2*nobs)
    cv_upper = llr - np.percentile(test_stats, 2.5, axis=0)*np.sqrt(omega2*nobs)

    return  2*(0 >= cv_upper) + 1*(0 <= cv_lower)

######################################################################################################
######################################################################################################
######################################################################################################

def ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,alpha=.05,nsims=1000,c=0):
    
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
    
    quant = lambda sig,c: np.quantile( np.abs( J_Lmod(sig,c)/np.sqrt(J_omod(sig,c))) ,1-alpha )

    sigstar = lambda c : minimize(lambda sig: -1*quant(sig[0],c), [2.5]).x
    cv0 = quant(sigstar(0),0) # critical value with c=0
    
    z_normal = norm.ppf(1-alpha/2)
    z_norm_sim = max(z_normal,np.quantile(np.abs(Z_L),1-alpha)) #simulated z_normal
    
    cv = max(cv0,z_normal)
    cstar = np.array([c])

    #Computing the ND test statistic:
    nLR_hat = ll1.sum() - ll2.sum()
    nomega2_hat = (ll1- ll2).var() ### this line may not be correct #####                    
    #Non-degenerate Vuong Tests    
    Tnd = (nLR_hat+V.sum()/2)/np.sqrt(n*nomega2_hat + cstar*(V*V).sum())
    
    return 1*(Tnd[0] >= cv) + 2*(Tnd[0] <= -cv)

######################################################################################################
######################################################################################################
######################################################################################################



def monte_carlo(total,gen_data,setup_shi,trials=100,use_boot2=False,c=0):
    reg = np.array([0, 0 ,0])
    boot1 = np.array([0, 0 ,0])
    boot2 = np.array([0, 0 ,0])
    boot3 = np.array([0, 0 ,0])
    shi = np.array([0, 0 ,0])
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
        reg_index = regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
        #boot_index1 = bootstrap_test_pt(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=c,trials=trials)
        c_star = 20*nobs
        boot_index1 = bootstrap_test_pt(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=c_star,trials=trials)
        boot_index2 = bootstrap_test_pt(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=c_star,trials=trials)
        boot_index3 = bootstrap_bc(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=c_star,trials=trials)
        shi_index = ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
        
        #update the test results
        reg[reg_index] = reg[reg_index] + 1
        boot1[boot_index1] = boot1[boot_index1] + 1
        boot2[boot_index2] = boot2[boot_index2] + 1
        boot3[boot_index3] = boot3[boot_index3] + 1
        shi[shi_index] = shi[shi_index] + 1

    return reg/total,boot1/total,boot2/total,boot3/total,shi/total,llr/total,np.sqrt( (var/total-(llr/total)**2) ),omega*np.sqrt(nobs)/total

def print_mc(mc_out):
    reg,boot1,boot2,boot3,shi, llr,std, omega = mc_out
    print('\\begin{tabular}{|c|c|c|c|c|c|}')
    print('\\hline')
    print('Model &  Normal & Bootstrap & Bootstrap-bc & Bootstrap-bc & Shi (2015) \\\\ \\hline \\hline')
    labels = ['No selection', 'Model 1', 'Model 2']
    for i in range(3): 
        print('%s & %.2f & %.2f & %.2f & %.2f & %.2f   \\\\'%(labels[i], reg[i],boot1[i],boot2[i],boot3[i],shi[i]))
    print('\\hline')
    print('\\end{tabular}')





