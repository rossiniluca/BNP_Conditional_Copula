function res= Gibbs_Cal(mg,mg1,x,PA)
%% Gibbs sampler algorithm and Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                 Written by LUCA ROSSINI                     %%%%%%%
%%%%%%%              Free University of Bozen, Italy                %%%%%%%
%%%%%%%            Ca' Foscari University of Venice, Italy          %%%%%%%
%%%%%%%             email address: luca.rossini@unibz.it            %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%##########################################################################
% Input
% mg,mg1 = overall measures of each twin's performance at school (y_1,y_2)
% x = covariate variable (mother's,father's level of education - family income)
% PA = (PA.burn, PA.save,PA.tot) 
%    = burn-in, saving and total number iterations for the Gibbs sampler
%%##########################################################################
% Output
% res = (res.Nclust,res.Dsave,res.alpsave,res.betsave,res.Wsave)
%     = posterior value for the number of clusters, allocation v., beta and weights
%%##########################################################################

%% Initialization
u = ksdensity(mg,mg,'function','cdf'); v = ksdensity(mg1,mg1,'function','cdf'); 
xx = norminv(u,0,1); yy = norminv(v,0,1);
n = size(u,1);
PA.kmax = 200; % initialization
PA.iter = 200; % n° of iterations to display
a = 1; b = 1; % prior on the weights ~ Beta(a,b)
% choice for the MH steps:
gg = 2;  % if gg==1 ==> as in Walker et al (2014)
         % if gg==2 ==> allocation variables change
NMH = 20; % n° of iterations for the MH
sigma = 0.5; % variance of the trunc normal of MH
med.dec = 2;  %if med.dec==1 ==> mean of rho after burn-in
              %if med.dec==2 ==> last rho
med.burn = 25;
% parameters for the truncated normal and uniform distributions
col.Min = 0.001; col.Max = 0.2;
CU.L = 0; CU.H = 2;
PR.Mu = 0; PR.Sg = 1000;
sce = 1; %if sce==1 ==> 1st calibration ft
         %if sce==2 ==> 2nd calibration ft
D = ones(n,1); % inizialize allocation variable
% save variables
W = zeros(1,PA.kmax);
res.Nclust = zeros(PA.save,1);
res.Dsave = zeros(n,PA.save);
res.alpsave = zeros(PA.kmax,PA.save);
res.betsave = zeros(PA.kmax,PA.save);
res.Wsave = zeros(PA.kmax,PA.save);
alp = zeros(1,PA.kmax); bet = zeros(1,PA.kmax);
alp(1,1) = random('unif',0,0.2); bet(1,1) = random('unif',0,0.2);

%% GIBBS SAMPLING
tic;
fprintf('Now Running...\n')
for irep=1:PA.tot
    % Update of the weights and stick-breaking components
    Dstar = max(D);
    Wcum = 0;
    for j=1:Dstar
        seleq = (D==j);
        aj = sum(seleq);
        selgr = (D>j);
        bj = sum(selgr);
        V(j) = random('beta',aj+a,bj+b,1);
        W(j) = prod(1-V(1:j-1))*V(j);
        Wcum = Wcum+W(j);
    end
    % Update of the slice variables
    Z = unifrnd(0,W(D));
    zstar = min(Z);
    Nstar = Dstar;
    while Wcum<(1-zstar)
        Nstar = Nstar+1;
        V(Nstar) = random('beta',a,b,1);
        W(Nstar) = prod(1-V(1:Nstar-1))*V(Nstar);
        Wcum = Wcum+W(Nstar);
    end
    % Update of the vector parameters  
    for k=1:Nstar
        if (sum(D==k)==0)
            pda = makedist('Normal','mu',alp(k),'sigma',sigma);
            ta = truncate(pda,col.Min,col.Max); alp(k) = random(ta,1,1);
            pdb = makedist('Normal','mu',bet(k),'sigma',sigma);
            tb = truncate(pdb,col.Min,col.Max); bet(k) = random(tb,1,1);
        else
            sD = (D==k); 
            alpold = alp(k); 
            betold = bet(k);
            [alp(k),bet(k)] = MHrho(u,v,x,alpold,betold,sD,gg,NMH,sigma,col,PR);
        end
    end
    % Update the allocation variables
    for t=1:n
        Dstart = find(W>Z(t));
        fco = condnormcopdens(u(t,:),v(t,:),x(t,:),alp(W>Z(t)),bet(W>Z(t)),0,0,sce);
        fco = fco/sum(fco);
        [tmp,sel] = histc(rand,[0;cumsum(fco')]);
        D(t) = Dstart(sel);
    end
    
    if mod(irep,PA.iter)==0
        fprintf('%4.0f out of %4.0f Iterations. \n', irep, PA.tot)
        toc;
    end
    % Save the variables
    if irep>PA.burn
        [tmp frq] = histc(D,(1:Nstar));
        res.Nclust(irep-PA.burn,1) = sum(tmp~=0);
        res.Dsave(:,irep-PA.burn) = D;
        res.alpsave(:,irep-PA.burn) = alp;
        res.betsave(:,irep-PA.burn) = bet;
        res.Wsave(:,irep-PA.burn) = W;
        res.Nstarsave(irep-PA.burn,1) = Nstar;
    end   
end