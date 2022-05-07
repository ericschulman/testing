overlap_shi_ex_pt_slow.ipynb
- main examples from paper, main reason slow is because i do analytical log likelihood... maybe to a version with correct guess?
- uses test v3, v3 does not actually re estimate the models in the bootstrap samples to save time.

1v1 
- trying to guage 1v1 regressor what happens...

shi_examples_summer_1_slow
- same as shi_examples_summer_1.ipynb. The code runs slower due to refitting model at each step...

shi_examples_summer_1_fast
- same as shi_examples_summer_1.ipynb. The code runs faster and I only fit the model 1x like vuong test 5. I demonstrate different versions of the test...

shi_examples_summer_1
- same as overlap_shi_ex_pt.ipynb. updates include v5 test. v5 test includes bias correction on regular test. 

shi_examples_summer_2
- same as overlap_shi_ex_pt_slow.ipynb. updates include v5 test. v5 test includes bias correction on regular test. 

kstat_table
- code to get the tables with the k-stats working
- actual examples
shi's examples:
a = 0 , k = 4, 9 ,19 - for some reasons
a = .25, k = 4, 9, 19 


kstat_table_2
- example with an actual overlapping regressor + spurious regressors?
- similar to edgeworth_reg2_pt?


edgeworth_shi_pt 
- examples from xiaoxia's paper

edgeworth_reg2_pt
- disorganized examples, includes the non-overlapping case, also includes code for doing the k-stats... will hopefully be outdated soon
