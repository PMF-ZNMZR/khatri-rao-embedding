function [Z,D,Y,res,niter,timings]=lr_gfadi_cplx(A,B,E,C,F,G,pA,pB,maxiter,rtol,tcrit)
% function [Z,D,Y,res,niter]=lr_gfadi_cplx(A,B,E,C,F,G,pA,pB,maxiter,rtol,tcrit)
%
% Generate a low rank matrices Z,Y and diagonal D such that X=Z*D*Y' solves the
% Sylvester equation: 
%        
% A*X*C - E*X*B = F*G'                              (1)
%
% The function implements the generalized low rank Sylvester-ADI method,
% see [1,2,3] (also called (g-)fADI).
%
% Inputs: 
% 
% A,B,E,C,F,G      The matrices F,G in the above equation.
%  p       a vector of shift parameters
% maxiter  maximum iteration number.
% rtol     tolerance for the residual norm based stopping criterion
% tcrit    (currently unused) structure defining termination criterion:

% Outputs:
%  
% Z,D,Y     The solution factors Z,D,Y such that Z*D*Y'=X solves (1).
% res       the vector of residuals 
%                   (1st row: direct comp, 2nd row: Lanczos)
% niter     the number of iteration steps taken
% timings   comp.times of various stages 
%               (1st row: full step, 2nd row: res.norm computation)
%
% [1] Benner/Li/Truhar, On the ADI method for Sylvester equations, J. Comput. Appl. Math.,
%                       346 233(4):1035–1045, 2009.
% [2] Benner/Kuerschner, Computing Real Low-rank Solutions of Sylvester equations by the Factored ADI
%                       Method, Comput. Math. Appl., 67(9):1656–1672, 2014.
% [3] Kuerschner, Inexact linear solves in the low-rank ADI iteration for
%      large Sylvester equations, Arxiv e-print 2312.02891,  2023

% Patrick Kuerschner, 2016-2024

% input parameters not fully checked!
if (nargin<11)||(isempty(tcrit)), tcrit=struct('nres',1,'relch',0,'nrmResD',0); end
% stopping criteria, currently only res.norm supported
% if tcrit.nres && ~tcrit.relch
    resi=1; relch=0;
    term = 'res(1,i)<rtol';
% elseif ~tcrit.nres && tcrit.relch
%     relch=1; resi=0;
%     term = 'rc(i)<sqrtneps';
% else
%     resi=1; relch=1;
%     term='(res(1,i)<rtol)||(rc(i)<sqrtneps)';
% end

%backward error type normalization of Lyap.residual
nrmResD=tcrit.nrmResD;
% various sizes & dimensions
n=size(A,1);
m=size(B,1);
r=size(F,2);
lA=length(pA);
lB=length(pB);
In=speye(n);
Im=speye(m);
Ir=speye(r);
% U=zeros;
tadi=tic;
%starting residual norm
res0=max(sqrt(real(eig((F'*F)*(G'*G)))));
if nrmResD
    nrmF=normest(A)*normest(C)+normest(B)*normest(E);
end

ac=pA(1);
bc=pB(1);
nrmZ=0;
Vt=F; Wt=G;
Z=[];Y=[];D=[];
res=zeros(2,maxiter);

rc=res;rc(1)=1; %unused
opts.isreal=false;
opts.issym=true;
sqrtneps=sqrt(n*eps);
timings=zeros(2,maxiter);
resLan=0;  %toggle res.norm comp. via Lanczos on/off
rL=0;
timings(1,1)=toc(tadi);
for i=1:maxiter %Sylvester-ADI Loop
   tadi=tic;
   %select pair of shifts
  ipA=mod(i+lA-1,lA)+1;
  ipB=mod(i+lB-1,lB)+1;
   ap=ac;
  ac=pA(ipA);
  bp=bc;
  bc=pB(ipB);
  % solve lin.sys.
  V1=(A-bc*E)\Vt;
  W1=(B-ac*C)'\Wt;
  %augment low-rank factor    
  Z=[Z V1];
  Y=[Y W1];
  D=blkdiag(D,(bc-ac)*Ir);
  %update residual factors
  Vt=Vt+(bc-ac)*E*V1; Wt=Wt+(ac-bc)'*C'*W1;
  
  % residual norm
  if resi
      denomR=res0;
      if nrmResD
          denomR=denomR+nrmF*norm(Y*D'*chol(Z'*Z,'lower'),2);
      end
      restime=tic;
      res(1,i)=max(sqrt(real(eig((Vt'*Vt)*(Wt'*Wt)))))/denomR;
      timings(2,i)=toc(restime);
      if resLan % resnorm via lanczos.
          restime=tic;
          res(2,i)=syl_r_norm(A,B,F,G,Z,D,Y,m,E,C,[],[])/denomR;
          timings(3,i)=toc(restime);
      end
  end

%   fprintf(1,['step: %4d  n.resLR: %2.4e resLan: %2.4e\n'],i,res(1,i),res(2,i));      

  %are we done?
  if eval(term)
    % fprintf('\n\n');
    timings(1,i)=timings(1,i)+toc(tadi);
    break;
  end
  timings(1,i)=timings(1,i)+toc(tadi);
end
niter=i;
timings=timings(:,1:niter);
res=res(:,1:niter);
rc=rc(:,1:niter); %unused