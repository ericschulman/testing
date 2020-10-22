%This is the log density function of model \mathcal{F} in the normal
%regression example . 
%
%The function takes two input variables:
%
%x:     the data vector: d_x\times 1 or d_x\times n
%theta: parameter of model \mathcal{F}
%
%The function produces at most three outputs:
%
%logfi:    \log(f(x,\theta))
%d_logfi:  \partial\log(f(x,\theta))/(\partial\theta)
%d2_logfi: \partial^2\log(f(x,\theta))/(\partial\theta^2): the second
%derivative matrix is vectorized into a row when the x input to the
%function has more than one rows.
%
%In the normal regression example, x=(y,z_1, z_2) where y is the dependent
%variable, z_1 is the regressor vector of the first model and z_2 is the
%regressor vector of the second model. The parameter \theta is d_{z_2}+2
%dimensional, which include the d_{z_2}+1 regression coefficients and one
%error variance.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [logfi,d_logfi,d2_logfi] = logfi(x,theta)
y = x(:,1);                       %dependent variable of the normal regression model

dz1 = length(theta)-2;            %dimension of nonconstant regressor

z1 = x(:,2:dz1+1);                %regressor

coeff = theta(1:dz1+1,1) ;          %regression coefficient
var = theta(dz1+2,1) ;             %error variance

nz = length(z1(:,1));             %number of rows of the data

zmat1 = [ones(nz,1),z1];
%size(zmat1);

logfi = -log(2*pi*var)/2-(y - zmat1*coeff).^2/(2*var);

if nargout>1;
    d_logfi = [repmat((y-zmat1*coeff),1,dz1+1).*zmat1/var...
                     -1/(2*var)+(y - zmat1*coeff).^2/(2*var^2)];
end

if nargout>2;
    d2_logfi = NaN(nz,(dz1+2)^2);
    for i = 1:nz;
        yi = y(i);
        zi = zmat1(i,:);
        dev2 = [-zi'*zi/var,-(yi-zi*coeff)*zi'/var^2;...
            -(yi-zi*coeff)*zi/var^2,...
            1/(2*var^2) - (yi - zi*coeff)^2/(var^3)];
        d2_logfi(i,:) = reshape(dev2,1,(dz1+2)^2);
    end
end
end

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



