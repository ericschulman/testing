import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm
from vuong_test_base import *



    

##############################
def choose_c(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,trials=500):
    
    #set up stuff
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    nobs = ll1.shape[0]
    
    #set c so the variance of the test stats is about omega?
    cstars = np.arange(0,16,2)
    cstars = 2**cstars 
    omegas = nobs*(ll1 - ll2).var() + cstars*(V*V).sum()
    cstar_results =  (omegas - nobs)**2
    c = cstars[cstar_results.argmin()]
    
    # return the cstar that makes omega =2?
    return c


def bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=0,trials=500,alpha=.05):
    nobs = ll1.shape[0]

    llr = ll1 -ll2
    test_stats = []
    bias_correct = []
    variance_stats  = []
     
    for i in range(trials):
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),nobs,replace=True)
        llrs = llr[sample]
        test_stats.append( llrs.sum() )
        variance_stats.append( llrs.var() )


    #final product, bootstrap
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    test_stats =  np.array(test_stats)
    variance_stats = np.array(variance_stats)
    test_statsnd = np.array(test_stats+ V.sum()/(2))
    
    #special situation for my test...
    stage1_distr = compute_stage1(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2)
    cutoff = np.percentile(stage1_distr, 100*(1-alpha), axis=0)
    variance_statsnd = np.clip(variance_stats,cutoff ,100000)
    #test_statsnd_var = test_statsnd.copy()
    #test_statsnd_var[variance_stats <= cutoff] = 0
    test_stats1,test_statsnd1,test_statsnd2 = (test_stats/variance_stats,
        test_statsnd/variance_stats, 
        test_statsnd/variance_statsnd)
    #print(variance_stats.min(),variance_statsnd.min(),cutoff)
    #print(np.percentile(test_statsnd,2.5),np.percentile(test_statsnd,97.5))
    #print(np.percentile(test_statsnd1,2.5),np.percentile(test_statsnd1,97.5))
    #print(np.percentile(test_statsnd2,2.5)-1/test_stats.size,np.percentile(test_statsnd2,97.5)+1/test_stats.size)
    #print('------')
    return test_stats,test_statsnd1,test_statsnd2 


 
def bootstrap_test(test_stats,nd=False,alpha=.05):
    cv_upper = np.percentile(test_stats, 100*(1-alpha/2) , axis=0)
    cv_lower = np.percentile(test_stats, 100*(1-alpha/2) , axis=0)
    if nd:
        cv_lower = cv_lower - 100*alpha/2/test_stats.size
        cv_upper = cv_upper +  100*alpha/2/test_stats.size
    return  2*(0 > cv_upper) + 1*(0 < cv_lower)


def monte_carlo(total,gen_data,setup_shi,trials=500,biascorrect=False):
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
        ll1,grad1,hess1,params1,ll2,grad2,hess2,params2 = setup_shi(yn,xn)
        llrn = (ll1 - ll2).sum()
        omegan = np.sqrt( (ll1 -ll2).var())
        llr = llr +llrn
        var = llrn**2 + var
        omega = omega +omegan
        
        #run the test
        reg_index = regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,biascorrect=biascorrect)
        twostep_index = two_step_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,biascorrect=biascorrect)
        shi_index,boot_index1,boot_index2,boot_index3 = 0,0,0,0 #take worst case for now...
        cstar = choose_c(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,trials=500)
        shi_index = ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)

        #bootstrap indexes....
        test_stats,test_statsnd1,test_statsnd2 = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=0,trials=trials)
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
    print('\\begin{tabular}{|c|c|c|c|c|c|c|}')
    print('\\hline')
    print('Model &  Normal & Two-Step & Bootstrap & Bootstrap-TIC & Bootstrap-ND & Shi (2015) \\\\ \\hline \\hline')
    labels = ['No selection', 'Model 1', 'Model 2']
    for i in range(3): 
        print('%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f   \\\\'%(labels[i], round(reg[i],2),round(twostep[i],2),round(boot1[i],2),round(boot2[i],2),round(boot3[i],2),round(shi[i],2)))
    print('\\hline')
    print('\\end{tabular}')





