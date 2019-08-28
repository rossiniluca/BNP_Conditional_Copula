function [aa,bb,alpold,betold]=MHrho(u,v,x,alpold,betold,sD,gg,NMH,sigma,col,PR)
%% M-H for the posterior of vector of beta = (beta_1,beta_2) for 1st calibration function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                 Written by LUCA ROSSINI                     %%%%%%%
%%%%%%%              Free University of Bozen, Italy                %%%%%%%
%%%%%%%            Ca' Foscari University of Venice, Italy          %%%%%%%
%%%%%%%             email address: luca.rossini@unibz.it            %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%##########################################################################
% Input
% u,v = inverse of the CDF of the observations u,v
% alphold,betold = (beta_1,beta_2) at the previous iterations
% sD = for allocation variable
% gg = can be 1 (if no allocation v.) or 2 (if with allocation v.)
% NMH = number of iterations for the MH
% col = made by col.Min, col.Max, min and max for the truncated distributions
% CU = lower and higher value of uniform distribution for the proposal
% PR = made by pri.mu,pri.sg mean and std deviation of the normal prior
%##########################################################################
% Output
% aa,bb = new value of (beta_1,beta_2)
% alpold,betold = vector of all (beta_1,beta_2) of the MH
%##########################################################################

n = size(u,1);
if gg==1
    xe = u; ye = v; ze = x;
    ni = n;
elseif gg==2
    xe = u(sD); ye = v(sD); ze = x(sD);
    ni = sum(sD);
end

for i=2:NMH
    % proposal as truncated normal distribution
    pda = makedist('Normal','mu',alpold(i-1,1),'sigma',sigma);
    ta = truncate(pda,col.Min,col.Max);
    alpstar=random(ta,1,1);
    pdb = makedist('Normal','mu',betold(i-1,1),'sigma',sigma);
    tb = truncate(pdb,col.Min,col.Max);
    betstar=random(tb,1,1);
    lognum1=log(truncatednormalpdf(alpold(i-1,1),alpstar,sigma,col.Min,col.Max))+...
        log(truncatednormalpdf(betold(i-1,1),betstar,sigma,col.Min,col.Max));
    logden1=log(truncatednormalpdf(alpstar,alpold(i-1,1),sigma,col.Min,col.Max))+...
        log(truncatednormalpdf(betstar,betold(i-1,1),sigma,col.Min,col.Max));

    %likelihood function for proposal
    tetstar=alpstar+betstar.*ze.^2;
    romstar=2./(abs(tetstar)+1)-1;
    pconstar=sqrt(1-romstar.^2);
    ipconstar=1./pconstar;
    prodconstar=prod(ipconstar);
    
    parespstar=-(1./(2*(1-romstar.^2))).*(romstar.^2.*(xe.^2+ye.^2)-2*romstar.*xe.*ye);
    espstar=exp(parespstar);
    suespstar=sum(espstar);
    %prior computations
    prstar=log(pdf('normal',alpstar,PR.Mu,PR.Sg))+log(pdf('normal',betstar,PR.Mu,PR.Sg));
    lognum=log(prodconstar)+suespstar+prstar;
    
    %likelihood function for alpold,betold
    tetold=alpold(i-1,1)+betold(i-1,1).*ze.^2;
    romold=2./(abs(tetold)+1)-1;
    pconold=sqrt(1-romold.^2);
    ipconold=1./pconold;
    prodconold=prod(ipconold);
    
    parespold=-(1./(2.*(1-romold.^2))).*(romold.^2.*(xe.^2+ye.^2)-2.*romold.*xe.*ye);
    espold=exp(parespold);
    suespold=sum(espold);
    %prior
    prold=log(pdf('normal',alpold(i-1,1),PR.Mu,PR.Sg))+log(pdf('normal',betold(i-1,1),PR.Mu,PR.Sg));
    logden=log(prodconold)+suespold+prold;
    
    alf=min([1,exp(lognum+lognum1-(logden+logden1))]);
    al(i,1)=alf;
    u=rand(1,1);
    
    if u<alf
        alpold(i,1)=alpstar;
        betold(i,1)=betstar;
    else
        alpold(i,1)=alpold(i-1,1);
        betold(i,1)=betold(i-1,1);
    end
    
end

aa=alpold(NMH,1);
bb=betold(NMH,1);