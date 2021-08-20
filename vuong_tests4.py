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


######################################################################################################
######################################################################################################
######################################################################################################

def ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,alpha=.05,nsims=1000):
    
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
    cstar = 0
    adapt_c = False

    if adapt_c:
        if cv0 - z_norm_sim > 0.5:  # if critical value with c=0 is not very big
            #set up array with cstars
            cstars = np.arange(0,16,2)
            cstars = 2**cstars - 1

            ##will loop through to find best...
            cstar_results = []
            cv_results = []

            for cstar in cstars:
                cv_result =  max(quant(sigstar(cstar),cstar),z_normal)
                cv_results.append(cv_result)
                cstar_results.append( (cv_result - z_norm_sim + .5)**2 )

            cstar_results = np.array(cstar_results)
            cv_results = np.array(cv_results)

            #set critical value and c_star?
            cv = cv_results[cstar_results.argmin()]
            cstar = cstars[cstar_results.argmin()]

    #Computing the ND test statistic:
    nLR_hat = ll1.sum() - ll2.sum()
    nomega2_hat = (ll1- ll2).var() ### this line may not be correct #####                    
    #Non-degenerate Vuong Tests    
    Tnd = (nLR_hat+V.sum()/2)/np.sqrt(n*nomega2_hat + cstar*(V*V).sum())
    return 1*(Tnd >= cv) + 2*(Tnd <= -cv)

#######################################################################
#######################################################################

def compute_stage1(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2):
    nsims = 5000
    
    k1 = params1.shape[0]
    k2 = params2.shape[0]
    k = k1 + k2
    
    V = compute_eigen2(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2)
    np.random.seed()
    Z0 = np.random.normal( size=(nsims,k) )**2
    
    return np.matmul(Z0,V*V)


def two_step_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,biascorrect=False):
    stage1_distr = compute_stage1(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2)
    nobs = ll1.shape[0]
    
    omega = np.sqrt( (ll1 -ll2).var())#set up stuff
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    llr = (ll1 - ll2).sum()
    if biascorrect:
        llr = llr + V.sum()/(2) #fix the test...
    test_stat = llr/(omega*np.sqrt(nobs))
    stage1_res = ( nobs*omega**2 >= np.percentile(stage1_distr, 95, axis=0) )    
    
    return (1*(test_stat >= 1.96) + 2*( test_stat <= -1.96))*stage1_res
    



######################################################################################################
######################################################################################################
######################################################################################################

def bootstrap_distr(yn,xn,nobs,model1,model2,setup_shi,trials=100):
    test_stats = []
    bias_correct = []
    variance_stats  = []

    for i in range(trials):
        subn = nobs
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),subn,replace=True)
        ys,xs = yn[sample],xn[sample]
        ll1,grad1,hess1,params1,model1,ll2,grad2,hess2,params2,model2 = setup_shi(ys,xs)
        
        ####messing around with recentering########
        V = compute_eigen2(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2)
        bias_correct.append(V.sum()/(2))
        ###################

        llr = (ll1 - ll2).sum() 
        omega2 = (ll1 - ll2).var()
        test_stats.append(llr)
        variance_stats.append((np.sqrt(omega2*nobs)))
        
    test_stats = np.array(test_stats)
    bias_correct = np.array(bias_correct)
    variance_stats= np.array(variance_stats) 

    variance_statsnd = np.clip(variance_stats,.1,100000)
    return (test_stats/variance_stats,
            (test_stats + bias_correct)/variance_stats, 
             (test_stats + bias_correct)/variance_statsnd)


def bootstrap_test(test_stats,nd=False):
    cv_upper = np.percentile(test_stats, 97.5, axis=0)
    cv_lower = np.percentile(test_stats, 2.5, axis=0)
    if nd:
        cv_lower = cv_lower - 10/test_stats.size
        cv_upper = cv_upper + 10/test_stats.size
    return  2*(0 >= cv_upper) + 1*(0 <= cv_lower)


def monte_carlo(total,gen_data,setup_shi,trials=100):

    reg = np.array([0, 0 ,0])
    twostep = np.array([0, 0 ,0])
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
        ll1,grad1,hess1,params1,model1,ll2,grad2,hess2,params2,model2 = setup_shi(yn,xn)
        llrn = (ll1 - ll2).sum()
        omegan = np.sqrt( (ll1 -ll2).var())
        llr = llr +llrn
        var = llrn**2 + var
        omega = omega +omegan
        
        #shi/twosteptest....
        reg_index = regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
        twostep_index = two_step_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
        shi_index = ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
        
        #bootstrap indexes....
        test_stats,test_statsnd1,test_statsnd2 = bootstrap_distr(yn,xn,nobs,model1,model2,setup_shi,trials=trials)
        boot_index1 = bootstrap_test(test_stats)
        boot_index2 = bootstrap_test(test_statsnd1)
        boot_index3 = bootstrap_test(test_statsnd2,nd=True)
        
        #update the test results
        reg[reg_index] = reg[reg_index] + 1
        twostep[twostep_index] = twostep[twostep_index] +1
        boot1[boot_index1] = boot1[boot_index1] + 1
        boot2[boot_index2] = boot2[boot_index2] + 1
        boot3[boot_index3] = boot3[boot_index3] + 1
        shi[shi_index] = shi[shi_index] + 1

    return reg/total,twostep/total,boot1/total,boot2/total,boot3/total,shi/total,llr/total,np.sqrt( (var/total-(llr/total)**2) ),omega*np.sqrt(nobs)/total



def print_mc(mc_out):
    reg,twostep, boot1,boot2,boot3,shi, llr,std, omega = mc_out
    print(boot1,boot2,boot3)
    print('\\begin{tabular}{|c|c|c|c|c|c|c|}')
    print('\\hline')
    print('Model &  Normal & Two-Step & Bootstrap & Bootstrap-TIC & Bootstrap-ND & Shi (2015) \\\\ \\hline \\hline')
    labels = ['No selection', 'Model 1', 'Model 2']
    for i in range(3): 
        print('%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f   \\\\'%(labels[i], round(reg[i],2),round(twostep[i],2),round(boot1[i],2),round(boot2[i],2),round(boot3[i],2),round(shi[i],2)))
    print('\\hline')
    print('\\end{tabular}')

    print('llr:%s, std:%s'%(llr,std))
