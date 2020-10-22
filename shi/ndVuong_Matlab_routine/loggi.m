%This is the log density function of model \mathcal{G} in the normal
%regression example . 
%
%The function takes two input variables:
%
%x:     the data vector: d_x\times 1 or d_x\times n
%beta: parameter of model \mathcal{G}
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

function [loggi,d_loggi,d2_loggi] = loggi(x,beta)
y = x(:,1);                       %dependent variable of the normal regression model

dz2 = length(beta)-2;             %dimension of nonconstant regressor

dx = length(x(1,:));              %dimension of the data vector

z2 = x(:,dx-dz2+1:dx);            %regressor - the last dz2 columns

coeff = beta(1:dz2+1);            %regression coefficient
var = beta(dz2+2);                %error variance

nz = length(z2(:,1));             %number of rows of the data

zmat2 = [ones(nz,1),z2];

loggi = -log(2*pi*var)/2- (y - zmat2*coeff).^2/(2*var);

if nargout>1;
    d_loggi = [repmat((y-zmat2*coeff),1,dz2+1).*zmat2/var...
        -1/(2*var)+(y - zmat2*coeff).^2/(2*var^2)];
end

if nargout>2;
    d2_loggi = NaN(nz,(dz2+2)^2);
    for i = 1:nz;
        yi = y(i);
        zi = zmat2(i,:);
        
        dev2 = [-zi'*zi/var,-(yi-zi*coeff)*zi'/var^2;...
            -(yi-zi*coeff)*zi/var^2,1/(2*var^2) - (yi - zi*coeff)^2/(var^3)];
        
        d2_loggi(i,:) = reshape(dev2,1,(dz2+2)^2);
    end
end
end

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



