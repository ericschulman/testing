import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm

######################################################################################################
######################################################################################################
######################################################################################################

def ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,alpha=.05,nsims=1000,adapt_c=True):
    
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
    cstar = 2

    if adapt_c:
        #print(z_norm_sim,cv0)
        if cv0 - z_norm_sim > 0.5:  # if critical value with c=0 is not very big
            #set up array with cstars
            cstars = np.linspace(0,.75,4)

            ##will loop through to find best...
            cstar_results = []
            cv_results = []

            for cstar in cstars:
                cv_result =  quant(sigstar(cstar),cstar)
                cv_results.append(cv_result)
                cstar_results.append( (cv_result - z_norm_sim )**2 )

            #print(z_norm_sim,cv0,cstars,cstar_results,cv_results)
            cstar_results = np.array(cstar_results)
            cv_results = np.array(cv_results)

            #set critical value and c_star?
            cv = cv_results[cstar_results.argmin()]
            cstar = cstars[cstar_results.argmin()]
            #print(cv,cstar)

    #Computing the ND test statistic:
    nLR_hat = ll1.sum() - ll2.sum()
    nomega2_hat = (ll1- ll2).var() ### this line may not be correct #####                    
    #Non-degenerate Vuong Tests    
    Tnd = (nLR_hat+V.sum()/2)/np.sqrt(n*nomega2_hat + cstar*(V*V).sum())
    return 1*(Tnd >= cv) + 2*(Tnd <= -cv)

#######################################################################
#######################################################################


def compute_test_stat(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2):

    nobs = ll1.shape[0]
    omega = np.sqrt((ll1 -ll2).var())
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    llr = (ll1 - ll2).sum()

    return llr,omega,V,nobs


def regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,alpha=.05, c=0,refinement_test=False,biascorrect=False):
    llr,omega,V,nobs = compute_test_stat(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)

    if biascorrect:
        llr = llr + V.sum()/(2) #fix the test...
    test_stat = llr/(omega*np.sqrt(nobs))

    refine_factor = 1
    if refinement_test:
        #print(omega,c,(V*V).sum(),np.sqrt(nobs),c*(V*V).sum()/np.sqrt(nobs))
        refine_factor =omega/(omega+c*(V*V).sum()/np.sqrt(nobs))
    return 1*(refine_factor*test_stat >= norm.ppf(1-alpha/2) ) + 2*( refine_factor*test_stat <= norm.ppf(alpha/2))


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
    

def compute_stage1(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2):
    nsims = 5000
    
    k1 = params1.shape[0]
    k2 = params2.shape[0]
    k = k1 + k2
    
    V = compute_eigen2(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2)
    np.random.seed()
    Z0 = np.random.normal( size=(nsims,k) )**2
    
    return np.matmul(Z0,V*V)


def two_step_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2, alpha=.05, biascorrect=False):
    stage1_distr = compute_stage1(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2)
    nobs = ll1.shape[0]
    
    omega = np.sqrt( (ll1 -ll2).var())#set up stuff
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    llr = (ll1 - ll2).sum()
    if biascorrect:
        llr = llr + V.sum()/(2) #fix the test...
    test_stat = llr/(omega*np.sqrt(nobs))
    stage1_res = ( nobs*omega**2 >= np.percentile(stage1_distr, 100 - alpha*100, axis=0) )    
    
    return (1*(test_stat >=  norm.ppf(1-alpha/2) ) + 2*( test_stat <=  norm.ppf(alpha/2) ))*stage1_res
    
    