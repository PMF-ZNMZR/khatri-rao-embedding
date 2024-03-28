function [H,V] = arn_pl(E, A, k,w)
%
%  Arnoldi method w.r.t. E\A
%
%  Calling sequence:
%
%    [H,V] = arn_pl(A,E,k,w)
%
%  Input:
%
%    A,E         the matrices A,E;
%    k         number of Arnoldi steps
%    w         initial vector (optional - chosen by random, if omitted).
%
%  Output:
%
%    H         matrix H ((k+1)-x-k matrix, upper Hessenberg);
%    V         matrix V (n-x-(k+1) matrix, orthogonal columns).
%

na = nargin;
n = size(A,1);                 % Get system order.

if k >= n-1, error('k must be smaller than the order of A!'); end
if na<4, w = randn(n,1); end 

V = zeros(n,k+1);
H = zeros(k+1,k);

V(:,1) = (1.0/norm(w))*w;

beta = 0;

for j = 1:k
 
  if j > 1
    H(j,j-1) = beta;
    V(:,j) = (1.0/beta)*w;
  end
  
  %mat.vec.-product
  w = A*V(:,j);
  if ~isempty(E)
    w=E \ w;
  end
  for it=1:2 %orthogonalization (twice)
      for i = 1:j
          coef = V(:,i)'*w;
          H(i,j) = H(i,j)+coef;
          w = w-coef*V(:,i);
      end
  end
  beta = norm(w);
  H(j+1,j) = beta;
 
end  

V(:,k+1) = (1.0/beta)*w;