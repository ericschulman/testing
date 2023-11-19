import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm

from vuong_test_base import *


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


def bootstrap_test(test_stats,nd=False,alpha=.05):
    cv_upper = np.percentile(test_stats, 100-alpha*50, axis=0)
    cv_lower = np.percentile(test_stats, alpha*50, axis=0)
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
