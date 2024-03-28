function [X, lam] = test_contour(namedparams)
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Testing routine for the contour integral method.

    arguments
        namedparams.example = 'gauss';
        namedparams.n = 100;
        namedparams.method = 'method0';
        namedparams.random_vectors_type = 'rank-one-multiply';
        namedparams.sylvester_solver = 'backslash';
        namedparams.sylv_put_shift_in_lyap = true; % Put the node z in the "Lyapunov" of the Sylvester eqn (and not as a separate term z*MXM of the generalized Sylvester equation)
        namedparams.n_quad_points = 40;
        namedparams.dim_right_projection = 6;
        namedparams.verbosity = 1;

        % Params for the biCGstab algorithm.
        namedparams.biCGstab_tol = 1e-8; 

        % If is_potential_separate=true, then K is always the discrete Laplacian. If false, then parts 
        % of the potential are incorporated into the K matrix.
        namedparams.is_potential_separate = false; 

        % Don't run the method, but compute the eigenvectors with eig and
        % plot their ranks. Only for small size problems.
        namedparams.just_analyze_true_solution = false;
    end

    config = struct();

    % Available examples: 
    % 'sum-of-squares', 'sum-of-squares-minus-xy', 'deal-ii', 'trefethen-exp', 'double-well', 'singular-one-over-sum-squares', 
    % 'discontinuous', 'gauss', 'matieu'
    config.example_name = namedparams.example;

    % Available methods:
    % 'method0'
    config.method_name = namedparams.method;

    config.problem_size = namedparams.n;  % Discretization size n (eigenvalue problem will be n^2 x n^2).

    % Available random_vector_type-s:
    % 'gaussian';          % Ordinary random Gaussian vectors.
    % 'rank-one-multiply'; % Khatri-Rao product of random Gaussian vectors, multiplied out (not stored as Kronecker factors).
    % 'rank-one';          % Khatri-Rao product of random Gaussian vectors, kept in factored form. Sylvester solver must support this.
    config.random_vectors_type = namedparams.random_vectors_type;

    % Available sylvester_solver-s:
    % config.sylvester_solver = 'backslash';
    % config.sylvester_solver = 'biCGstab';
    config.sylvester_solver = namedparams.sylvester_solver;

    % sylv_put_shift_in_lyap is true or false.
    % If true: 
    %   Put the shift z (i.e. the integration node) in the "Lyapunov" of the Sylvester eqn: (z/2*M-K)*XM + MX(z/2*M-K) - VR*X*VL = wR*wL'.
    % If false:
    %   Put the shift z a separate term z*MXM of the generalized Sylvester eqn: -KXM - MXK - VR*X*VL + zMXM = wR*wL'.
    config.sylv_put_shift_in_lyap = namedparams.sylv_put_shift_in_lyap; 

    % Params for the biCGstab algorithm.
    config.biCGstab_tol = namedparams.biCGstab_tol; 
    
    % If is_potential_separate=true, then K is always the discrete Laplacian. If false, then parts 
    % of the potential are incorporated into the K matrix.
    config.is_potential_separate = namedparams.is_potential_separate; 

    config.n_quad_points = namedparams.n_quad_points;
    config.dim_right_projection = namedparams.dim_right_projection;  % Number of columns in the random matrix used for projection on the right.

    % Set verbosity to 0 for silence, and increase for more output.
    config.verbosity = namedparams.verbosity;
    logprintf( set_verbosity = config.verbosity );
       
    % Do we run the method, or only compute and analyze the true solution with eigs?
    if( namedparams.just_analyze_true_solution )
        example = generate_example( config.example_name, n=config.problem_size );
        analyze_true_solution( example );
    else
        [X, lam, example] = run( config );
        if( config.verbosity > 0 )
            evaluate_solution( example, config, X, lam );
        end
    end
end


function [X, lam, example] = run( config )
    logprintf( 1, 'Preparing example ''%s''...', config.example_name );
    example = generate_example( config.example_name, n=config.problem_size, is_potential_separate=config.is_potential_separate );
    % example = generate_example( config.example_name, n=config.problem_size, output_mode='full' ); % Use this with examples that cannot be generated in the factored form
    logprintf( 1, 'done.\nEigenvalue problem size = %d.\n\n', example.size );

    logprintf( 1, 'Running method ''%s''...\n', config.method_name );
    logprintf( 2, '--------------------------------------------------------------\n' );
    t_start = tic;

    switch( config.method_name )
        case 'method0'
            [X, lam] = method0( example, config );
        
        otherwise
            error( 'Unknown method.' );
    end

    t_end = toc(t_start);
    logprintf( 2, '--------------------------------------------------------------\n' );
    logprintf( 1, 'Finished running method ''%s'' in %.2f seconds.\n\n', config.method_name, t_end );
end


function evaluate_solution( example, config, X, lam )
% Print and plot some stats about the exact and the computed solution.
    
    logprintf( 1, 'Solver properties:\n' );
    logprintf( 1, '\t- using random vectors of type:  ''%s''\n', config.random_vectors_type );
    logprintf( 1, '\t- number of quadrature points:   %d\n', config.n_quad_points );
    logprintf( 1, '\t- right projection to dimension: %d\n', config.dim_right_projection );
    logprintf( 1, '\n' );

    [X_true, lam_true] = analyze_true_solution( example );
    n_true = length( lam_true );

    logprintf( 1, 'Analyzing computed solution:\n' );
    res = eigenpair_residuals( example, config, X, lam );
    idx_found = ( abs(lam - example.center) < example.radius );
    res = res(idx_found);

    logprintf( 1, '\tComputed %d eigenvalues; ', length(lam) );
    logprintf( 1, '%d eigenvalues are inside the contour. \n', length(res) );

    if( example.is_symmetric )
        imag_max = max( abs( imag( lam(idx_found) ) ) );
        logprintf( 1, '\tTruncating computed eigs to real: max imag part = %.4e\n', imag_max );
        lam_computed = real( lam( idx_found ) ); 
    else
        lam_computed = lam(idx_found);
    end

    [~, P] = sort( real(lam_computed) );
    lam_computed = lam_computed(P);
    X_computed = X(:, idx_found ); 
    X_computed = X_computed(:, P);
    res = res(P);

    n_computed = length( lam_computed );
    logprintf( 1, '\n\tFound %d eigenvalues inside the contour (there are %d actually):\n', n_computed, n_true );

    if( example.is_symmetric )
        logprintf( 1, '\t\t computed       |  true           | residual   | lam_true-lam_computed\n' );
        for i = 1:max(n_computed, n_true)
            if( i <= length(lam_true) && i <= length(lam_computed) )
                logprintf( 1, '\t\t% .8e | % .8e | %.4e | %.4e\n', lam_computed(i), lam_true(i), res(i), abs( lam_true(i)-lam_computed(i)) );
            elseif( i <= length(lam_computed) )
                logprintf( 1, '\t\t% .8e |                 | %.4e\n', lam_computed(i), res(i) ); 
            else
                logprintf( 1, '\t\t                | % .8e | \n', lam_true(i) );
            end
        end
    else 
        logprintf( 1, '\t\t computed               |  true                   | residual     | lam_true-lam_computed\n' );
        for i = 1:max(n_computed, n_true)
            if( i <= length(lam_true) && i <= length(lam_computed) )
                logprintf( 1, '\t\t% .4e %+.4e | % .4e %+.4e | %.4e | %.4e \n', real(lam_computed(i)), imag(lam_computed(i)), real(lam_true(i)), imag(lam_true(i)), res(i), abs( lam_true(i)-lam_computed(i)) );
            elseif( i <= length(lam_computed) )
                logprintf( 1, '\t\t% .4e %+.4e |                        | %.4e\n', real(lam_computed(i)), imag(lam_computed(i)), res(i) ); 
            else
                logprintf( 1, '\t\t                       | % .4e %+.4e | \n', real(lam_true(i)), imag(lam_true(i)) );
            end
        end
    end

    logprintf( 1, '\n\n' );

    figure; hold on;
    plot(real(lam_true),imag(lam_true),'x','markersize',10,'linewidth',2);
    plot(real(lam_computed),imag(lam_computed),'o','markersize',10,'linewidth',2);
    title('Computed and true eigenvalues');
    legend('true eigs', 'computed eigs');
    hold off;
end


function [X_true, lam_true] = analyze_true_solution( example )
    logprintf( 1, 'Analyzing true solution:\n' );

    A = example.A;
    B = example.B;

    % Compute true eigenvalues and eigenvectors.
    [X_true, D] = eigs( A, B, example.num_eigs+5, example.center );    
    n = sqrt( length(A) );

    lam_true = diag(D);
    P = ( abs(lam_true - example.center) < example.radius );
    lam_true = lam_true(P);
    X_true = X_true(:, P);
    [~, P] = sort( real(lam_true) );
    lam_true = lam_true(P);
    X_true = X_true(:, P);
    n_true = length( lam_true );

    logprintf( 1, '\tTrue eigenvalues inside the contour:\n' );
    for i = 1 : n_true
        if( isreal(lam_true(i)))
            logprintf( 1, '\t\t%.8e\n', lam_true(i) );
        else
            logprintf( 1, '\t\t%.8e %+.8ei\n', real(lam_true(i)), imag(lam_true(i)) );
        end
    end
    
    V = X_true;

    figure();
    surf( example.V ); shading interp;
    title( 'Potential' );

    figure();
    V_all = [];
    logprintf( 1, '\tEigenvector ranks:\n' );
    for i = 1 : min(n_true, 4)
        V1 = reshape(V(:,i), n, n);
        V_all = [V_all V1];

        logprintf( 1, '\t\t%d\n', rank(V1) );
        subplot(220+i);
        surf(real(V1)); shading interp
    end
    title( sprintf( 'First %d eigenvectors (real parts)', min(n_true, 4) ) );

    logprintf( 1, '\tRank of [V1, V2, V3, V4] where Vj = mat(eigvec):\n' );
    logprintf( 1, '\t\t%d\n', rank( V_all ) );

    figure();
    nsqrt = sqrt( example.size );
    X_true_all = []; X_true_all_label = '';
    for i = 1:n_true
        semilogy( svd( reshape(X_true(:,i), nsqrt, nsqrt ) ), '-', 'DisplayName', sprintf('eigenvector U_%d', i), 'LineWidth', 2 );
        hold on;

        X_true_all = [X_true_all, reshape(X_true(:,i), nsqrt, nsqrt )];
        X_true_all_label = sprintf( '%s U_{%d}', X_true_all_label, i );
    end

    semilogy( svd( X_true_all ), 'k--', 'DisplayName', sprintf('[%s]', X_true_all_label), 'LineWidth', 2 );

    title('Singular values of the folded eigenvectors' );
    legend('show');
    xlim([0, 100]);
    % save_picture('figure3_left.pdf');
    hold off;

    for i = 1:n_true
        ss = svd( reshape(X_true(:,i), nsqrt, nsqrt) );
        logprintf( 1, '\tTrue eigvec %d -> rank=%d, sing. values = %.4e %.4e %.4e %.4e\n', ...
            i, rank( reshape(X_true(:,i), nsqrt, nsqrt) ), ss(1), ss(2), ss(3), ss(4) );
    end

    logprintf( 1, '\n' );
end
