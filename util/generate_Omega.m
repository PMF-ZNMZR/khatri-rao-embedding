function Omega = generate_Omega( n, ell, mode )
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Generates an n x ell random matrix Omega.
% If mode = 'gauss':
%   - All entries are random Gaussian.
% If mode = 'rank-one': 
%   - Omega is a Khatri-Rao product of two random Gaussian matrices.
%   - Omega is kept in factored form, as as struct with fields Omega.left and Omega.right (Omega = khatri_rao(Omega.left, Omega.right)). 
% If mode = 'rank-one-multiply':
%   - Omega is a Khatri-Rao product of two random Gaussian matrices.
%   - Columns in Omega are multiplied out, i.e. Omega is an ordinary matrix.

    nsqrt = sqrt( n );

    % Generate the random matrix.
    switch( mode )
        case 'gaussian'
            % Use ordinary Gaussian random vectors.
            Omega = randn(n, ell);
            
        case 'rank-one'
            % Generate Omega as a matrix of random low rank vectors. Keep it factored.
            Omega = struct();
            Omega.left  = randn(nsqrt, ell);
            Omega.right = randn(nsqrt, ell);

        case 'rank-one-multiply'
            % Generate Omega as a matrix of random low rank vectors. Multiply out the columns.
            left  = randn(nsqrt, ell);
            right = randn(nsqrt, ell);

            Omega = [];
            for i = 1:ell
                Omega = [Omega kron(left(:, i), right(:, i))];
            end

        otherwise
            error( 'generate_Omega :: unknown mode' );
    end
end
