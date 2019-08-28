function cc=condnormcopdens(u,v,xx,alp,bet,gam,delta,sce)
%% Conditional Gaussian/normal density of the copula
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                 Written by LUCA ROSSINI                     %%%%%%%
%%%%%%%              Free University of Bozen, Italy                %%%%%%%
%%%%%%%            Ca' Foscari University of Venice, Italy          %%%%%%%
%%%%%%%             email address: luca.rossini@unibz.it            %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%##########################################################################
% Input
% (u,v) = observations
% xx = conditional variable
% alp,bet = (beta_1,beta_2)
% sce = can be 1 (1st calibration ft) or 2 (2nd calibration ft)
%##########################################################################
% Output
% cc = copula density function
%##########################################################################

x = norminv(u,0,1);
y = norminv(v,0,1);

%conditional components
if sce==1
    thet = alp+bet*xx^2;
elseif sce==2
    thet = alp+bet*xx+gam.*exp(-delta*xx^2);
end


rho = (2./(abs(thet)+1))-1;

cov = 1-rho.^2;

cc = cov.^(-0.5).*exp(-0.5.*(rho.^2.*(x.^2+y.^2)-2.*rho.*x.*y).*cov.^(-1));