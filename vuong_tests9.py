import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm
from vuong_test_base import *



    

##############################
def bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c1=.01,c2=.02,trials=500,alpha=.05,print_stuff=False,data_tuned_c=False):
    llr,omega,V,nobs = compute_test_stat(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    #tuning param thing
    stage1_distr = compute_stage1(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2)
    tuning_cv = np.sqrt(np.percentile(stage1_distr, 100 - alpha*100, axis=0))
    tuning_cv_min = np.sqrt(np.percentile(stage1_distr, alpha*100, axis=0))

    LR = llr/nobs #make this easier to figure out
    #modifications to c1 
    c1_mod = c1 #*np.sqrt( (V*V).sum()/nobs )

    test_stat1 = np.sqrt(nobs)*LR/omega
    test_stat2 = np.sqrt(nobs)*(LR + V.sum()/(2*nobs))/omega
    test_stat3 = np.sqrt(nobs)*(LR + V.sum()/(2*nobs))/(omega+c1_mod)

    llr_full = (ll1 -ll2)
    test_stats = []
    bias_correct = []
    variance_stats  = []
    
    for i in range(trials):
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),nobs,replace=True)
        llrs = llr_full[sample]
        test_stats.append( llrs.sum() )
        variance_stats.append( llrs.var() )

    #final product, bootstrap
    test_stats =  np.array(test_stats)/nobs
    test_statsnd = np.array(test_stats+ V.sum()/(2*nobs))

    variance_stats = np.sqrt(variance_stats)
    variance_statsnd = variance_stats + c1_mod

    #original results...
    test_stats1,test_stats2 = (np.sqrt(nobs)*test_stats/(variance_stats) - test_stat1,
        np.sqrt(nobs)*test_statsnd/(variance_stats)- test_stat2)

    
    #setting up nd test with assymetric bootstrap
    test_stats3 = np.sqrt(nobs)*test_statsnd/(variance_statsnd)
    
    if data_tuned_c is not None:
        distr_term = np.quantile(stage1_distr, np.clip(data_tuned_c,0,1) )
        #distr_term = np.sqrt(distr_term) #, accidentally suqaring... this stuff seems like a really good way to check if omega is growign tho...
        c2 = c2*distr_term

    bonus_term = np.sqrt(nobs)*c2/(omega+c1_mod)
    if print_stuff:
        #what if we used quantiles of omega to pick?
        quantiles = [0.01, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99]
        # Calculate the quantiles
        quantile_values = np.quantile(stage1_distr, quantiles)
        quantile_values = np.sqrt(quantile_values)
        # Print all quantiles on one line
        print("Quantiles:", ", ".join(f"{int(q * 100)}th: {value:.2f}" for q, value in zip(quantiles, quantile_values)))

        #print other stuff
        #print('bonus term',np.sqrt(nobs)*c2/(omega+c1_mod),'// variance stats',np.sqrt(nobs)*omega, '//part1', (nobs)*c2, '//part2', np.sqrt(nobs)*(omega+c1_mod), np.sqrt(nobs)*omega  )
        print('bonus term',np.sqrt(nobs)*c2/(omega+c1_mod),'// variance stats',np.sqrt(nobs)*omega, '//part1', np.sqrt(nobs)*c2/c1_mod, '//part2', 1- omega/(omega+c1_mod)   )

    test_stats3a,test_stats3b = test_stats3 -test_stat3 + bonus_term, test_stats3 -test_stat3 -bonus_term

    return test_stats1,test_stats2,test_stats3a ,test_stats3b ,test_stat1,test_stat2,test_stat3


 
def bootstrap_test(test_stats,test_stat,alpha=.05,print_stuff=False,left=True,right=True):
    cv_upper = np.percentile(test_stats, 100*(1-alpha/2) , axis=0)
    cv_lower = np.percentile(test_stats, 100*(alpha/2) , axis=0)
    if print_stuff:
        print(cv_lower,test_stat,cv_upper)
        if  left:
            print('---')
    return  1*(test_stat > cv_upper)*right + 2*(test_stat < cv_lower)*left


def monte_carlo(total,gen_data,setup_shi,skip_boot=False,skip_shi=False,refinement_test=False,trials=500,biascorrect=False,
        c1=None,c2=None,alpha=.05,adapt_c=True,print_stuff=True,data_tuned_c=None):
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
            shi_index = ndVuong(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,alpha=alpha,adapt_c=adapt_c)

        #bootstrap indexes....
        boot_index1,boot_index2,boot_index3 = 0,0,0
        if not skip_boot:
            print_stuff_i =  ((i%25) == 0 ) and print_stuff

            test_stats1,test_stats2,test_stats3a,test_stats3b ,test_stat1,test_stat2,test_stat3 = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c1=c1,c2=c2,trials=trials,print_stuff=print_stuff_i,data_tuned_c=data_tuned_c)
            #print('-- test1 --')
            boot_index1 = bootstrap_test(test_stats1,test_stat1,alpha=alpha)
            #print('-- test2 --')
            boot_index2 = bootstrap_test(test_stats2,test_stat2,alpha=alpha,print_stuff=False)
            #print('-- test3--')

            
            boot_index3a = bootstrap_test(test_stats3a,test_stat3,alpha=alpha,print_stuff=print_stuff_i,left=False) #test right side
            boot_index3b = bootstrap_test(test_stats3b,test_stat3,alpha=alpha,print_stuff=print_stuff_i,right=False) #test left side
            boot_index3 = max(boot_index3a,boot_index3b)
            #print(boot_index3a,boot_index3b,boot_index3,'----')
        #update the test results
        reg[reg_index] = reg[reg_index] + 1
        twostep[twostep_index] = twostep[twostep_index] +1
        refine_test[refine_index] = refine_test[refine_index] +1
        boot1[boot_index1] = boot1[boot_index1] + 1
        boot2[boot_index2] = boot2[boot_index2] + 1
        boot3[boot_index3] = boot3[boot_index3] + 1
        shi[shi_index] = shi[shi_index] + 1

    
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

def print_mc2(mc_out):
    reg,twostep, refine_test, boot1,boot2,boot3,shi, llr,std, omega = mc_out
    print('\\begin{tabular}{|c|c|c|}')
    print('\\hline')
    print('Model &  Normal & Bootstrap-ND  \\\\ \\hline \\hline')
    labels = ['No selection', 'Model 1', 'Model 2']
    for i in range(3): 
        print('%s & %.2f & %.2f \\\\'%(labels[i], round(refine_test[i],2),round(boot3[i],2) ))
    print('\\hline')
    print('\\end{tabular}')







