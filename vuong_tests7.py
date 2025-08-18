import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm
from vuong_test_base import *



    

##############################



def bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c1=.01,c2=.02,trials=500,alpha=.05):
    llr,omega,V,nobs = compute_test_stat(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    #tuning param thing
    stage1_distr = compute_stage1(ll1,grad1,hess1,params1,ll2, grad2,hess2,params2)
    tuning_cv = np.sqrt(np.percentile(stage1_distr, 100 - alpha*100, axis=0))
    tuning_cv_min = np.sqrt(np.percentile(stage1_distr, alpha*100, axis=0))
    test_stat1 = llr/(omega*np.sqrt(nobs))
    test_stat2 = (llr + V.sum()/(2))/(omega*np.sqrt(nobs))
    test_stat3 = (llr + V.sum()/(2))/((omega+c1*(V*V).mean() )*np.sqrt(nobs))

    llr_full = ll1 -ll2
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
    test_stats =  np.array(test_stats)
    test_statsnd = np.array(test_stats+ V.sum()/(2))

    test_statsnda = np.array(test_stats+ V.sum()/2+tuning_cv*c2)
    test_statsndb = np.array(test_stats+ V.sum()/2-tuning_cv*c2)
    if c1 == .11:
        test_statsnda = np.array(test_stats+ V.sum()/2+tuning_cv*c2/(omega**2))
        test_statsndb = np.array(test_stats+ V.sum()/2-tuning_cv*c2/(omega**2))

    if c1 ==.12:

        #clip the nuemrator?
        test_stats3a = np.clip(test_statsnd,  -1*tuning_cv, 1e10)
        test_stats3b = np.clip(test_statsnd, -1e10, tuning_cv)

        #modify test statistic with clipping too
        test_stat3 = np.clip( (llr + V.sum()/(2)) , -1*tuning_cv, tuning_cv) #clip the numerator
        test_stat3 = test_stat3/((omega+c1*(V*V).mean() )*np.sqrt(nobs))


    variance_stats = np.sqrt(variance_stats)
    variance_statsnd = variance_stats+ c1*(V*V).mean()

    if c1 == .1 or c1==.11:
        variance_statsnd = np.clip(variance_stats,.1,1000)
    #print('var//varnd//right//left',variance_stats.mean(),variance_statsnd.mean(),test_statsnda.mean(),test_statsndb.mean())

    #print('thing',c1*(V*V).mean())
    test_stats1,test_stats2,test_stats3a,test_stats3b = (test_stats/(variance_stats*np.sqrt(nobs)) - test_stat1,
        test_statsnd/(variance_stats*np.sqrt(nobs))- test_stat2, 
        test_statsnda/(variance_statsnd*np.sqrt(nobs))- test_stat3,
        test_statsndb/(variance_statsnd*np.sqrt(nobs))- test_stat3)
    #print('stuff',omega,variance_stats.mean(),omega+c1*(V*V).sum(),variance_statsnd.mean())
    




    return test_stats1,test_stats2,test_stats3a ,test_stats3b ,test_stat1,test_stat2,test_stat3


 
def bootstrap_test(test_stats,test_stat,alpha=.05,print_stuff=False,left=True,right=True):
    cv_upper = np.percentile(test_stats, 100*(1-alpha/2) , axis=0)
    cv_lower = np.percentile(test_stats, 100*(alpha/2) , axis=0)
    if print_stuff:
        print(cv_lower,test_stat,cv_upper)
        if  left:
            print('---')
    return  1*(test_stat > cv_upper)*right + 2*(test_stat < cv_lower)*left


def monte_carlo(total,gen_data,setup_shi,skip_boot=False,skip_shi=False,refinement_test=False,trials=500,biascorrect=False,c1=None,c2=None,alpha=.05,adapt_c=True,print_stuff=True):
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
            test_stats1,test_stats2,test_stats3a,test_stats3b ,test_stat1,test_stat2,test_stat3 = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c1=c1,c2=c2,trials=trials)
            #print('-- test1 --')
            boot_index1 = bootstrap_test(test_stats1,test_stat1,alpha=alpha)
            #print('-- test2 --')
            boot_index2 = bootstrap_test(test_stats2,test_stat2,alpha=alpha,print_stuff=False)
            #print('-- test3--')

            print_stuff_i =  ((i%25) == 0 ) and print_stuff
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







