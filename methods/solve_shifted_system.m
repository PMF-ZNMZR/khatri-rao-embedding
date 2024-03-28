function [Z, varargout] = solve_shifted_system( example, config, z, Omega ) 
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% For the pair (A, B) stored in example, return Z = (z*B-A) \ Omega.
%
    switch( config.sylvester_solver )
        case 'backslash'
            Z = solve_shifted_system_backslash( example, config, z, Omega );
        
        case 'biCGstab'
            [Z, hist] = solve_shifted_system_iterative( 'biCGstab', example, config, z, Omega );

        otherwise
            error( 'Unknown config.sylvester_solver' );
    end    

    if( nargout > 1 )
        varargout{1} = hist;
    end
end


function Z = solve_shifted_system_backslash( example, config, z, Omega )
    assert( ~isstruct(Omega), 'At this point Omega must be stored as a matrix, and not in the factored form. Use random_vectors_type=''rank-one-multiply''.' );

    Z = (z*example.B - example.A) \ Omega;
end


function [Z, hist] = solve_shifted_system_iterative( sylvester_solver, example, config, z, Omega )
    % A = kron(K, M)+kron(M, K) + \sum kron(Vleft{i}, Vright{i})
    % B = kron(M, M)
    % To solve (zB - A)Z = Omega, for each column kron(wL, wR) of Omega we solve a generalized Sylvester equation
    %       z*MXM - KXM - MXK - \sum Vright{i}*X*Vleft{i} = wR*wL', 
    % i.e.:
    %       (z/2*M-K)*XM + MX(z/2*M-K) - \sum Vright{i}*X*Vleft{i} = wR*wL', when config.sylv_shift_in_lyap == true
    % or,
    %       -KXM - MXK - \sum Vright{i}*X*Vleft{i} + zMXM = wR*wL', when config.sylv_shift_in_lyap == false
    % We use several sylvester_solver's:
    %   'biCGstab' -> with (linear) ADI preconditioner, as described in [1].
    %
    % [1] Benner, Breiten - Low rank methods for a class of generalized Lyapunov equations and related issues (2013)

    assert( isstruct(Omega), 'At this point Omega must be stored as a struct, in the factored form. Use random_vectors_type=''rank-one''.' );

    max_iters = 0;

    % Prepare parameter matrices for:
    % A1*X*B1' + A2*X*B2' + N1{1}*X*N2{1}' + ... + N1{l}*X*N2{l}' + C1*C2' = 0
    for j = 1 : size(Omega.left, 2)
        switch( config.sylv_put_shift_in_lyap )
            case true
                % Put z*MXM in to the Lyapunov part (i.e., into A1 together with K).
                A1 = z/2*example.M - example.K;
                B1 = example.M';
                A2 = example.M;
                B2 = A1'; % Note: z complex -> this is NOT the same as B2 = A1 !!!

                N1 = cell(1, length(example.Vright));
                for i = 1 : length(example.Vright)
                    N1{i} = -example.Vright{i};                    
                end
                N2 = example.Vleft;

                C1 = -Omega.right(:, j); % Note C1*C2' is on the left-hand side.
                C2 = Omega.left(:, j);

            case false
                % Do not put z*MXM in the Lyapunov part (i.e., put it in a separate N1{end+1}*N2{end+1}).
                A1 = -example.K;
                B1 = example.M';
                A2 = example.M;
                B2 = A1';

                N1 = cell(1, length(example.Vright));
                for i = 1 : length(example.Vright)
                    N1{i} = -example.Vright{i};                    
                end
                N2 = example.Vleft;

                N1{end+1} = z*example.M;
                N2{end+1} = example.M';

                C1 = -Omega.right(:, j); % Note C1*C2' is on the left-hand side.
                C2 = Omega.left(:, j);
        end

        % Solve.
        switch( sylvester_solver )
            case 'biCGstab'
                [Z.left{j}, Z.right{j}, hist] = sylv_multiterm_bicgstab( A1, B1, A2, B2, C1, C2, N1, N2, config );
                max_iters = max(max_iters, hist.niters);

            otherwise
                error( 'Unknown config.sylvester_solver' );
        end
    end

    switch( sylvester_solver )
        case 'biCGstab'
            logprintf( 1, '{%d}', max_iters );
    end
end
