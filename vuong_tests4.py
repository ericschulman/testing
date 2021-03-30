import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm

def regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2):
    llr = (ll1 - ll2).sum()
    nobs = ll1.shape[0]

    omega = np.sqrt( (ll1 -ll2).var())
    test_stat = llr/(omega*np.sqrt(nobs))
    return 1*(test_stat >= 1.96) + 2*( test_stat <= -1.96)


######################################################################################################
######################################################################################################
######################################################################################################

def ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,alpha=.05,nsims=1000):
    k1 = params1.shape[0]
    k2 = params2.shape[0]

    k = k1 + k2
    n = len(ll1)

    hess1 = hess1/n
    hess2 = hess2/n
    
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
    
    #diagonostic line
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
    cstar = np.array([0])
    
    #Computing the ND test statistic:
    nLR_hat = ll1.sum() - ll2.sum()
    nomega2_hat = (ll1- ll2).var() ### this line may not be correct #####                    
    #Non-degenerate Vuong Tests    
    Tnd = (nLR_hat+V.sum()/2)/np.sqrt(n*nomega2_hat + cstar*(V*V).sum())

    return 1*(Tnd[0] >= cv) + 2*(Tnd[0] <= -cv)

######################################################################################################
######################################################################################################
######################################################################################################


def bootstrap_distr_eic(yn,xn,nobs,model1,model2,setup_shi,trials=100):
    test_stats_eic = []
    test_stats= []

    for i in range(trials):
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),nobs,replace=True)
        ys,xs = yn[sample],xn[sample]
        ll1b,grad1b,hess1b,params1b,ll2b,grad2b,hess2b,params2b  = setup_shi(ys,xs)
        llrb = (ll1b - ll2b ).sum()
        test_stats.append( llrb )
        ll1b_eic, ll2b_eic = model1.loglike(params1b), model2.loglike(params2b)
        test_stats_eic.append( ( ll1b_eic - ll2b_eic ) )
    return np.array(test_stats),np.array(test_stats_eic)



def bootstrap_test(test_stats):
    cv_upper = np.percentile(test_stats, 97.5, axis=0)
    cv_lower = np.percentile(test_stats, 2.5, axis=0)
    return  2*(0 >= cv_upper) + 1*(0 <= cv_lower)


def bootstrap_pivot(ll1,ll2,test_stats,test_stats_eic):
    print('llr:%s, eic:%s, test_stat-mean:%s'%((ll1-ll2).sum(),test_stats_eic.mean(),test_stats.mean()))
    print('eic-med:%s, test_stat-med:%s'%((np.percentile(test_stats_eic, 50, axis=0),np.percentile(test_stats, 50, axis=0))))
    test_stat = (ll1-ll2).sum() - (test_stats.mean() - test_stats_eic.mean())
    cv_lower = 2*test_stat - np.percentile(test_stats_eic, 97.5, axis=0)
    cv_upper = 2*test_stat -  np.percentile(test_stats_eic, 2.5, axis=0)
    return  2*(0 >= cv_upper) + 1*(0 <= cv_lower)


def monte_carlo(total,gen_data,setup_shi,trials=100):

    reg = np.array([0, 0 ,0])
    boot1 = np.array([0, 0 ,0])
    boot2 = np.array([0, 0 ,0])
    shi = np.array([0, 0 ,0])

    omega = 0
    llr = 0
    var = 0

    for i in range(total):
        
        #setup data
        np.random.seed()
        yn,xn,nobs = gen_data()
        
        #update llr and summary stats
        ll1,grad1,hess1,params1,model1,ll2,grad2,hess2,params2,model2 = setup_shi(yn,xn,return_model=True)
        llrn = (ll1 - ll2).sum()
        llr = llr +llrn
        var = llrn**2 + var
        reg_index = regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
        shi_index = ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)

        #update test results
        print(ll1.sum())
        print(ll2.sum())
        print('-------')
        test_stats,test_stats_eic = bootstrap_distr_eic(yn,xn,nobs,model1,model2,setup_shi,trials=trials)

        #run the bootstrap test test...
        boot_index1 = bootstrap_test(test_stats_eic)
        boot_index2 = bootstrap_pivot(ll1,ll2,test_stats,test_stats_eic)
        
        #update test results...
        reg[reg_index] = reg[reg_index] + 1        
        boot1[boot_index1] = boot1[boot_index1] + 1
        boot2[boot_index2] = boot2[boot_index2] + 1
        shi[shi_index] = shi[shi_index] + 1

    return reg/total,boot1/total,boot2/total,shi/total,llr/total,np.sqrt(var/total-(llr/total)**2)



def print_mc(mc_out):
    reg,boot1,boot2, shi, llr,std = mc_out
    print('\\begin{tabular}{|c|c|c|c|c|}')
    print('\\hline')
    print('Model &  Normal & Bootstrap & Bootstrap-pt & Shi (2015) \\\\ \\hline \\hline')
    labels = ['No selection', 'Model 1', 'Model 2']
    for i in range(3): 
        print('%s & %.2f & %.2f & %.2f & %.2f   \\\\'%(labels[i], reg[i],boot2[i],boot1[i],shi[i]))
    print('\\hline')
    print('\\end{tabular}')
    print('\n')

    print('llr:%s, std:%s'%(llr,std))
