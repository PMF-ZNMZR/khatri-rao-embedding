function [X, lam] = method0( example, config )
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Implementation of the contour integral method.
%
% Solve the eigenvalue problem Ax = lam*Bx in the following way:
% 1. Compute Z = r(B^(-1)A)*B^(-1/2)*Omega using the quadrature formula, with the terms using (z_j B - A)^(-1) * B^(1/2) * Omega.
% 2. Compute Z = QR.
% 3. Compute Rayleigh quotients Ar = Q'*A*Q, Br = Q'*B*Q.
% 4. Compute Ritz pairs for (Ar, Br).
% 5. Report those Ritz values that are inside the contour.

    switch( config.sylvester_solver )
        case 'backslash'
            logprintf( 1, '\tSylvester solver: backslash\n' );
            % Tolerance for low-rank recompression of the columns in the solution.
            tol_recompress = 1e-8;

        case 'biCGstab'
            logprintf( 1, '\tSylvester solver: biCGstab, tol = %.2e\n', config.biCGstab_tol );
            % Tolerance for low-rank recompression of the columns in the solution.
            tol_recompress = config.biCGstab_tol / 100;
    end

    n = example.size;
    nsqrt = sqrt( n );

    % Generate the random matrix Omega.
    rnd_state = rng(); rng(123458); 
    Omega = generate_Omega( n, config.dim_right_projection, config.random_vectors_type );
    rng(rnd_state);

    % Quad points and weights on the circle.
    nc = config.n_quad_points;
    w = exp( 2i*pi*(1:nc)/nc );
    z = example.center + example.radius*w;
    
    % Quadrature.
    t_integration = tic; t = 0;
    Z = init_Z( Omega ); % Initialize Z to zero matrix.
    for k = 1:nc
        tic;
        if( config.verbosity >= 1 )
            if( k == 1 )
                logprintf( 1, '\tProcessing quad node ...' );
            end

            if( k > 1 )
                logprintf( 1, '(%.2fs)', t );
            end

            if( mod(k, 10) == 0 )
                logprintf( 1, '\n\tProcessing quad node ...' );
            end

            logprintf( 1, ' %d', k );

            if( k == nc )
                logprintf( 1, ' ...done.\n' );
            end
        end

        Z_node = solve_shifted_system( example, config, z(k), Omega );

        % Z = Z + example.radius*w(k)/nc * Z_node;
        Z = expand_Z( Z, example.radius, w(k), nc, Z_node, tol_recompress );
        
        if( isstruct( Z_node ) )
            logprintf( 2, 'quad node %d -> sylv. solution has %d cols, accumulated sum has %d cols.\n', k, size(Z_node.left{1}, 2), size(Z.left{1}, 2 ) );
        end

        if( config.verbosity >= 2 && nsqrt <= 1000 && mod(k, 10) == 0 )
            if( isstruct( Z_node ) )
                s = svd( reshape(Z_node.left{1} * Z_node.right{1}', nsqrt, nsqrt) );
                ind = min( find( s ./ s(1) < 1e-5) );

                logprintf( 2, 'quad node %d -> sylv. solution has %d cols; first col has sigma(%2d)/sigma(1) = %.4e\n', ...
                    k, size(Z_node.left{1}, 2), ind, s(ind)/s(1) );
            else
                % Rank of the first col in Z_node.
                s = svd( reshape(Z_node(:, 1), nsqrt, nsqrt) );
                ind = min( find( s ./ s(1) < 1e-5) );
                logprintf( 2, 'quad node %d -> first column of sylv. solution: sigma(%2d)/sigma(1) = %.4e\n', ...
                    k, ind, s(ind)/s(1) );
            end
        end
        t = toc;
    end
    t_integration = toc(t_integration);
    logprintf( 1, '\n\tTotal integration time: %.2fs\n', t_integration );

    % Recompress the columns if using a low-rank solver.
    if( isstruct( Z ) )
        for j = 1 : length( Z.left )
            [Z.left{j}, Z.right{j}, ~] = low_rank_truncate( Z.left{j}, Z.right{j}, tol_recompress );
        end
    
        % Ranks of cols in Z after the quadrature.
        logprintf( 1, '\tAccumulated sum for the first col in Omega has rank %d.\n', size( Z.left{1}, 2 ) );
    end

    % Compute and extract eigenvalues.
    % First, project (A, B) onto Z.
    if( isstruct( Z ) )
        logprintf( 1, 'Projecting to the final subspace, using two SVDs.\n' );

        % Collect all left/right factors, and compute their common basis.
        Z_mat = cell2mat( Z.left );
        [Q, s, ~] = svd( Z_mat, 0 );
        s = [diag(s); 0];
        ind = min( find( s ./ s(1) < tol_recompress) ); ind = min(ind, size(Q, 2));
        Q_left = Q(:, 1:ind);
        r_left = size( Q_left, 2 );

        logprintf( 3, '\tsize(Z_mat) = %d x %d\n', size(Z_mat, 1), size(Z_mat, 2));

        Z_mat = cell2mat( Z.right );
        [Q, s, ~] = svd( Z_mat, 0 );
        s = [diag(s); 0];
        ind = min( find( s ./ s(1) < tol_recompress) ); ind = min(ind, size(Q, 2));
        Q_right = Q(:, 1:ind);
        r_right = size( Q_right, 2 );

        logprintf( 1, '\tRank of the projection matrices r_left = %d, r_right = %d\n', r_left, r_right );

        % A = kron(K, M)+kron(M, K) + kron(VL, VR)
        % B = kron(M, M)
        % project onto the subspace spanned by kron(Q_right, Q_left) --> dimension is r_left*r_right instead of ell!
        A_proj = kron( Q_right'*(example.K*Q_right), Q_left'*(example.M*Q_left) ) + kron( Q_right'*(example.M*Q_right), Q_left'*(example.K*Q_left) ) + kron( Q_right'*(example.VL*Q_right), Q_left'*(example.VR*Q_left) );
        B_proj = kron( Q_right'*(example.M*Q_right), Q_left'*(example.M*Q_left) );

        A_proj = full( A_proj );
        B_proj = full( B_proj );

        logprintf( 3, '\tsize(A_proj) = %d x %d\n', size(A_proj, 1), size(A_proj, 2) );

        % Next, compute the QR-factorization of the (r_left*r_right)*l matrix whose columns are vectorized projections of Z.left{j}*Z.right{j}' in the basis kron(Q_left, Q_right).
        ell = config.dim_right_projection;
        Q = zeros( r_left*r_right, ell );
        for j = 1 : ell 
            ZZ_left = Q_left'*Z.left{j};
            ZZ_right = Q_right'*Z.right{j};
            
            Q(:, j) = reshape( ZZ_left*ZZ_right', r_left*r_right, 1 );
        end

        [Q, ~] = qr( Q, 0 );

        % Finally, obtain the projection from the l-dim. subspace.
        A_proj = Q'*A_proj*Q;
        B_proj = Q'*B_proj*Q;
    else
        [Q, ~] = qr( Z, 0 );
        A_proj = Q' * (example.A * Q);
        B_proj = Q' * (example.B * Q);
    end

    % Compute the Ritz values = eigenvalue approximations.
    [X_proj, lam] = eig( A_proj, B_proj );
    lam = diag( lam );

    % Normalize the eigenvectors of the projection.
    for j = 1 : config.dim_right_projection
        X_proj(:, j) = X_proj(:, j) / norm( X_proj(:, j) );
    end

    % Lift the eigenvectors;
    if( isstruct( Z ) )
        X = zeros( n, config.dim_right_projection );
        for j = 1 : config.dim_right_projection
            X_proj_mat = reshape( Q*X_proj(:, j), r_left, r_right );
            X(:, j) = reshape( Q_left*X_proj_mat*Q_right', n, 1 );
        end
    else   
        X = Q * X_proj;
    end
end


function Z = init_Z( Omega )
    if( isstruct( Omega ) )
        % Z is stored in a low-rank format.
        for j = 1 : size(Omega.left, 2)
            Z.left{j} = [];
            Z.right{j} = [];
        end
    else
        % Z is stored as a full matrix.
        Z = zeros( size( Omega ) );
    end
end


function Z = expand_Z( Z, radius, w, nc, Z_node, tol_recompress )
    % Z = Z + example.radius*w(k)/nc * Z_node;
    if( isstruct( Z_node ) )
        for j = 1 : length(Z.left)
            Z.left{j} = [Z.left{j} radius*w/nc*Z_node.left{j}];
            Z.right{j} = [Z.right{j} Z_node.right{j}];
        
            % Recompress.
            [Z.left{j}, Z.right{j}, ~] = low_rank_truncate( Z.left{j}, Z.right{j}, tol_recompress );
        end
    else 
        Z = Z + radius*w/nc * Z_node;
    end
end


function [Z1, Z2, norm_Z1Z2] = low_rank_truncate( Z1, Z2, tol )
    % Truncates of a low-rank matrix Z1*Z2'.
    % [Z1, Z2, norm_Z1Z2] = low_rank_truncate(Z10, Z20, tol) returns matrices Z1, Z2 such that
    %   || Z1*Z2' - Z10*Z20' ||_F <= tol*|| Z10*Z20' ||_F,
    % where the rank of Z1*Z2' is the smallest integer such that the above inequality holds. 
    % Also returns || Z10*Z20' ||_F.
        
    [Q1, R1] = qr( Z1, 0 );
    [Q2, R2] = qr( Z2, 0 );

    R1R2t = R1*R2';
    
    [U, S, V] = svd( R1R2t, "econ" );
    sigma = [diag(S); 0];

    norm_Z1Z2 = sqrt( sum(sigma.^2) );
    for r = 1:length(sigma)-1
        if( sqrt(sum(sigma(r+1:end).^2)) < tol * norm_Z1Z2 )
            break;
        end
    end

    Z1 = Q1 * U(:, 1:r) * diag( sqrt(sigma(1:r)) );
    Z2 = Q2 * V(:, 1:r) * diag( sqrt(sigma(1:r)) );

    norm_Z1Z2 = sqrt(sum(sigma(1:r).^2));
end
