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
    return max(c,2)


def bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c1=.01,c2=.02,trials=500,alpha=.05):
    llr,omega,V,nobs = compute_test_stat(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    test_stat1 = llr/(omega*np.sqrt(nobs))
    test_stat2 = (llr + V.sum()/(2))/(omega*np.sqrt(nobs))
    test_stat3 = (llr + V.sum()/(2))/((omega+c1*(V*V).sum())*np.sqrt(nobs))


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
    test_stats =  np.array(test_stats)
    test_statsnd = np.array(test_stats+ V.sum()/(2))

    variance_stats = np.sqrt(variance_stats)
    variance_statsnd = variance_stats+ c2*(V*V).sum()

    test_stats1,test_stats2,test_stats3 = (test_stats/(variance_stats*np.sqrt(nobs)) - test_stat1,
        test_statsnd/(variance_stats*np.sqrt(nobs))- test_stat2, 
        test_statsnd/(variance_statsnd*np.sqrt(nobs))- test_stat3)
    #print('stuff',omega,variance_stats.mean(),omega+c1*(V*V).sum(),variance_statsnd.mean())

    return test_stats1,test_stats2,test_stats3 ,test_stat1,test_stat2,test_stat3


 
def bootstrap_test(test_stats,test_stat,alpha=.05,print_stuff=False):
    cv_upper = np.percentile(test_stats, 100*(1-alpha/2) , axis=0)
    cv_lower = np.percentile(test_stats, 100*(alpha/2) , axis=0)
    if print_stuff:
        print(cv_upper,cv_lower,test_stat)
    return  2*(test_stat > cv_upper) + 1*(test_stat < cv_lower)


def monte_carlo(total,gen_data,setup_shi,skip_boot=False,skip_shi=False,refinement_test=False,trials=500,biascorrect=False,c1=None,c2=None,alpha=.05):
    reg = np.array([0, 0 ,0])
    twostep = np.array([0, 0 ,0])
    refine_test = np.array([0, 0 ,0])
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

        if c1 is None:
            stage1_distr = compute_stage1(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2)
            c1 = np.percentile(stage1_distr, 100 - alpha*100, axis=0)
            c2 = .1

        #run the test
        reg_index = regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,biascorrect=biascorrect,alpha=alpha)
        twostep_index = two_step_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,biascorrect=biascorrect,alpha=alpha)
        refine_index = regular_test(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,biascorrect=biascorrect,refinement_test=True,c=c1,alpha=alpha)
        shi_index,boot_index1,boot_index2,boot_index3 = 0,0,0,0 #take worst case for now...

        shi_index=0
        if not skip_shi:
            shi_index = ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,alpha=alpha)

        #bootstrap indexes....
        boot_index1,boot_index2,boot_index3 = 0,0,0
        if not skip_boot:
            test_stats1,test_stats2,test_stats3 ,test_stat1,test_stat2,test_stat3 = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c1=c1,c2=c2,trials=trials)
            #print('-- test1 --')
            boot_index1 = bootstrap_test(test_stats1,test_stat1,alpha=alpha)
            #print('-- test2 --')
            boot_index2 = bootstrap_test(test_stats2,test_stat2,alpha=alpha)
            #print('-- test3--')
            boot_index3 = bootstrap_test(test_stats3,test_stat3,alpha=alpha)

        #update the test results
        reg[reg_index] = reg[reg_index] + 1
        twostep[twostep_index] = twostep[twostep_index] +1
        refine_test[refine_index] = refine_test[refine_index] +1
        boot1[boot_index1] = boot1[boot_index1] + 1
        boot2[boot_index2] = boot2[boot_index2] + 1
        boot3[boot_index3] = boot3[boot_index3] + 1
        shi[shi_index] = shi[shi_index] + 1
        #print('-------------------------------')
    
    if refinement_test:
        return reg/total,twostep/total,refine_test/total,boot1/total,boot2/total,boot3/total,shi/total,llr/total,np.sqrt( (var/total-(llr/total)**2) ),omega*np.sqrt(nobs)/total
    
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





