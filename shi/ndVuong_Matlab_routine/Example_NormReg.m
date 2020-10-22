%This is a simple example based on the Example 1 of Shi (2013). The purpose
%of the example is to illustrate how to use the routine ndVuongcv.m
%
%In this example, we consider the following two models:
%
% F: Y = theta_1 + Z1*theta_2 + u, u\sim N(0,sigma_f^2)
% G: Y = beta_1 + Z2*beta_2 + v, v\sim N(0,sigma_g^2)
%
%The DGP is:
%
% Y = 1 + (a1/sqrt(d_F-2))*sum(Z1) + (a2/sqrt(d_G-2))*sum(Z2) + e
%
%     where
%        (Z1,Z2,e)\sim N(0,I)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
n = 250;    %sample size
Scv = 5000;  %Monte Carlo repetition

alpha = 0.05; %significance level

a1 = 0.25;
a2 = 0.25;

d_F = 3; %dimension of model F
d_G = 11; %dimension of model G
k = d_F+d_G;

[sample_rstr,cv_rstr]=RandStream.create('mrg32k3a','NumStreams',2);

%Create the fake data:
Z1 = randn(sample_rstr,n,d_F-2);
Z2 = randn(sample_rstr,n,d_G-2);
er = randn(sample_rstr,n,1);
Y = 1+ (a1/sqrt(d_F-2))*sum(Z1,2)+(a2/sqrt(d_G-2))*sum(Z2,2)+er;
    
data = [Y,Z1,Z2];  %The whole data matrix put together
    
%Parameter Estimators: (In general, should numerically maximize
%log-likelihood function. But here the problem is simple enough to have
%closed form solution. I use closed form solution here.)
Z1cons = [ones(n,1),Z1];
Z2cons = [ones(n,1),Z2];

theta_hat = (Z1cons'*Z1cons)\(Z1cons'*Y);       %coefficient estimators
beta_hat = (Z2cons'*Z2cons)\(Z2cons'*Y);
varf_hat = (Y-Z1cons*theta_hat)'*(Y-Z1cons*theta_hat)/n; %variance estimator
varg_hat = (Y-Z2cons*beta_hat)'*(Y-Z2cons*beta_hat)/n;


%critical value and test statistic
[Test_statistic,critical_value,c_star]=ndVuong('logfi','loggi',data,...
                [theta_hat;varf_hat],[beta_hat;varg_hat],alpha,cv_rstr,Scv)



