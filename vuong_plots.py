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
    nobs = len(ll1)

    hess1 = hess1/nobs
    hess2 = hess2/nobs

    k1,k2 = params1.shape[0],params2.shape[0]
    k = k1 + k2
    
    #A_hat:
    A_hat1 = np.concatenate([hess1,np.zeros((k2,k1))])
    A_hat2 = np.concatenate([np.zeros((k1,k2)),hess2])
    A_hat = np.concatenate([A_hat1,A_hat2],axis=1)
    Q_hat = np.concatenate([A_hat1,-1*A_hat2],axis=1)

    #B_hat, covariance of the score...
    B_hat = np.concatenate([grad1,-1*grad2],axis=1) 
    B_hat = np.cov(B_hat,rowvar=False)

    #compute eigenvalues for weighted chisq
    sqrt_B_hat= linalg.sqrtm(B_hat)
    W_hat = np.matmul(sqrt_B_hat,linalg.inv(Q_hat))
    W_hat = np.matmul(W_hat,sqrt_B_hat)
    #print(W_hat)
    V,W = np.linalg.eig(W_hat)


    #try another way to make sure I understand
    S_hat = linalg.inv(A_hat).dot(B_hat).dot( linalg.inv(A_hat))
    #print(S_hat)
    #print(Q_hat.dot(S_hat))
    #print(np.linalg.eig(Q_hat.dot(S_hat))[0].sum()/2)

    return V.astype(float)



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
    plt.hist( true_stats , density=True,bins=15, label="True",alpha=.75)
    return true_stats


def plot_analytic(yn,xn,nobs,setup_shi):
    n_sims = 5000
    model_eigs = compute_eigen(yn,xn,setup_shi)
    eigs_tile = np.tile(model_eigs,n_sims).reshape(n_sims,len(model_eigs))
    normal_draws = stats.norm.rvs(size=(n_sims,len(model_eigs)))
    weighted_chi = ((normal_draws**2)*eigs_tile).sum(axis=1)
    plt.hist(weighted_chi,density=True,bins=15,alpha=.75,label="Analytic")
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
    
    plt.hist( test_stats, density=True,bins=15, label="Bootstrap",alpha=.75)
    return test_stats

############################  actual test stat ########################

def plot_true2(gen_data,setup_shi,trials=500):
    true_stats = []
    for i in range(trials):
        np.random.seed()
        ys,xs,nobs = gen_data()
        nobs = ys.shape[0]
        ll1,grad1,hess1,k1,ll2, grad2,hess2,k2 = setup_shi(ys,xs)
        llr = (ll1 - ll2).sum()
        omega2 = (ll1 - ll2).var()
        true_stats.append(llr/(np.sqrt(omega2*nobs)))

    true_stats= np.array(true_stats)
    true_stats = true_stats - true_stats.mean()
    plt.hist( true_stats, density=True,bins=15, label="True",alpha=.75)
    return true_stats

def plot_analytic2(yn,xn,nobs,setup_shi):
    overlap,normal =  compute_analytic(yn,xn,setup_shi)
    plt.hist(overlap,density=True,bins=15,alpha=.75,label="Overlapping")
    plt.hist(normal,density=True,bins=15,alpha=.75,label="Normal")
    return overlap,normal

###################################

def plot_bootstrap_pt(yn,xn,nobs,setup_shi,trials=500,c=0):
    ll1,grad1,hess1,k1,ll2,grad2,hess2,k2 = setup_shi(yn,xn)
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
    V = compute_eigen2(ll1,grad1,hess1,k1,ll2,grad2,hess2,k2)
    test_stats = np.array(test_stats)
    test_stats  = test_stats - test_stats.mean()
    variance_stats = np.sqrt(variance_stats)*np.sqrt(nobs)

    plt.hist( test_stats/variance_stats, density=True,bins=15, label="Bootstrap",alpha=.75)
    return test_stats/variance_stats


def plot_bootstrap_recenter(yn,xn,nobs,setup_shi,trials=500,c=0):
    test_stats = []
    
    ############ messing around with recentering ###################
    ll1,grad1,hess1,ll2,theta1, grad2,hess2,theta2 = setup_shi(yn,xn)
    #need true "parameters" ...
    #######################################
    b = 0
    for i in range(trials):
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),nobs,replace=True)
        ys,xs = yn[sample],xn[sample]
        ll1b,grad1b,hess1b,theta1b,ll2b, grad2b,hess2b,theta2b  = setup_shi(ys,xs)
        
        ####messing around with recentering########
        theta_diff1 =  np.array([(theta1 - theta1b)])
        b1 = np.dot(theta_diff1,-1*hess1b/nobs)
        b1 = np.dot(b1,theta_diff1.transpose())

        theta_diff2 =  np.array([(theta2 - theta2b)])
        b2 = np.dot(theta_diff2,-1*hess2b/nobs)
        b2 = np.dot(b2,theta_diff2.transpose())

        b = b + nobs/2*(b1 -b2)
        ###################
        
        llrb = (ll1b - ll2b ).sum()
        llrb =  llrb - nobs/2*(b1 - b2)[0,0]
  
        omega2b = (ll1b - ll2b).var()
        test_stats.append(llrb)
        
    print(b/trials, np.array(ll1-ll2).sum() )
    test_stats = np.array(test_stats)
    plt.hist( test_stats/test_stats.std(), density=True,bins=15, label="Bootstrap",alpha=.75)
    return test_stats




def plot_bootstrap2(yn,xn,nobs,setup_shi,trials=500,c=0):
    test_stats = []
    

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
        omega2 = (ll1 - ll2).var() + c*V.sum()/(nobs)
        test_stats.append(llr/(np.sqrt(omega2*nobs)))
        

    plt.hist( test_stats, density=True,bins=15, label="Bootstrap",alpha=.75)
    return test_stats
