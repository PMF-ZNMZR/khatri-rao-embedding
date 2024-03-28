function res = eigenpair_residuals( example, config, X, lam ) 
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% For the pair (A, B) stored in example, return the residuals for eigenpair approximations (lam, X).
% For now, do not use any info about the structure in A, B.
% This is done only in the phase of verifying the final solution, so it may be inefficient.

    res = zeros( size(lam) );
    for i = 1 : length( lam )
        res(i) = norm( example.A * X(:, i) - lam(i)*(example.B * X(:, i)) );
    end
end
