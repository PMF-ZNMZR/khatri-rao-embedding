function Z = test_sylvester( namedparams )
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Testing routine for multiterm Sylvester solvers.

    arguments
        namedparams.example = 'sum-of-squares-minus-xy';
        namedparams.n = 300;
        namedparams.random_vectors_type = 'rank-one-multiply';
        namedparams.sylvester_solver = 'backslash';
        namedparams.sylv_put_shift_in_lyap = true;
        namedparams.dim_right_projection = 1;
        namedparams.verbosity = 4;
        namedparams.z_angle = pi/4;

        % Params for the biCGstab algorithm.
        namedparams.biCGstab_tol = 1e-8; 

        % If is_potential_separate=true, then K is always the discrete Laplacian. If false, then parts 
        % of the potential are incorporated into the K matrix.
        namedparams.is_potential_separate = false; 

        % Just compare rank-1 random vectors vs Gaussian random vectors, plot singular value decay.
        namedparams.just_compare_rank1_vs_gaussian = false;
    end

    config = struct();

    % Available examples: 
    % 'sum-of-squares', 'sum-of-squares-minus-xy', 'deal-ii', 'trefethen-exp', 'double-well', 'singular-one-over-sum-squares', 
    % 'discontinuous', 'gauss', 'matieu'
    config.example_name = namedparams.example;

    config.problem_size = namedparams.n;  % Discretization size n (eigenvalue problem will be n^2 x n^2).

    % Number of random vectors in Omega. Only 1 is currently supported.
    config.dim_right_projection = namedparams.dim_right_projection;
    assert( config.dim_right_projection == 1 );

    % Available random_vector_type-s:
    % 'gaussian';          % Ordinary random Gaussian vectors.
    % 'rank-one-multiply'; % Khatri-Rao product of random Gaussian vectors, multiplied out (not stored as Kronecker factors).
    % 'rank-one';          % Khatri-Rao product of random Gaussian vectors, kept in factored form. Sylvester solver must support this.
    config.random_vectors_type = namedparams.random_vectors_type;

    % Available sylvester_solver-s:
    % config.sylvester_solver = 'backslash';
    % config.sylvester_solver = 'biCGstab';
    config.sylvester_solver = namedparams.sylvester_solver;

    % Whether to put shifts into the Lyapunov part of the generalized Sylvester eqn (true), or into the "N"-part (false).
    config.sylv_put_shift_in_lyap = namedparams.sylv_put_shift_in_lyap;

    % If is_potential_separate=true, then K is always the discrete Laplacian. If false, then parts 
    % of the potential are incorporated into the K matrix.
    config.is_potential_separate = namedparams.is_potential_separate; 

    % Params for the biCGstab algorithm.
    config.biCGstab_tol = namedparams.biCGstab_tol; 
    
    example = generate_example( config.example_name, n=config.problem_size, is_potential_separate=config.is_potential_separate );
    % example = generate_example( example_name, n=config.problem_size, output_mode='full' ); % Use this with examples that cannot be generated in the factored form

    % Generate a point on the contour from the example.
    shift = example.center + example.radius * exp( 1i * namedparams.z_angle );

    % Set verbosity to 0 for silence, and increase for more output.
    logprintf( set_verbosity=namedparams.verbosity );

    % Solve the shifted system: (shift * B - A)*Z = Omega.
    % Equivalently, solve the Sylvester eqn with B = kron(M, M), A = kron(M, K) + kron(K, M) + kron(VL, VR).
    % Generate the random right-hand side for the Sylvester solver. Always generate the same RHS.
    % Generate rank-one vectors.
    rnd_state = rng(); rng(12345); 
    Omega = generate_Omega( example.size, config.dim_right_projection, config.random_vectors_type );
    rng(rnd_state);

    % Analyze the problem itself.
    config.verbosity = namedparams.verbosity;
    if( config.verbosity > 0 )
        analyze_sylvester_problem( example, config, shift, Omega, namedparams );
        logprintf( 1, 'Using Sylvester solver: ''%s''\n', config.sylvester_solver );
        logprintf( 1, '----------------------------------------\n' );
    end

    if( namedparams.just_compare_rank1_vs_gaussian )
        assert( strcmp( config.random_vectors_type, 'rank-one-multiply' ) );
        assert( strcmp( config.sylvester_solver, 'backslash' ) );

        % Solve the shifted system: (shift * B - A)*Z = Omega.
        % First with Omega = rank-1.
        Z_rank1 = solve_shifted_system( example, config, shift, Omega );
    
        % Generate unstructured random vectors.
        rnd_state = rng(); rng(12345); 
        Omega = generate_Omega( example.size, config.dim_right_projection, 'gaussian' );
        rng(rnd_state);

        Z_gauss = solve_shifted_system( example, config, shift, Omega );

        % Plot the singular values.
        nsqrt = namedparams.n;
        Z_rank1 = reshape( Z_rank1, nsqrt, nsqrt );
        Z_gauss = reshape( Z_gauss, nsqrt, nsqrt );

        figure();
        semilogy( svd(Z_rank1), 'DisplayName', '$\omega = \tilde{\omega} \otimes \hat{\omega}$', 'LineWidth', 2 );
        hold on;
        semilogy( svd(Z_gauss), 'DisplayName', '$\omega$ unstructured', 'LineWidth', 2 );
        title( 'Singular values of folded x = (zI - A)^{-1}\omega' );
        legend( 'show', 'Interpreter','latex' );
        % save_picture( 'figure3_right.pdf' );
    else 
        % Solve the shifted system: (shift * B - A)*Z = Omega.
        % Equivalently, solve the Sylvester eqn with B = kron(M, M), A = kron(M, K) + kron(K, M) + kron(VL, VR).
        Z = solve_shifted_system( example, config, shift, Omega );

        % If using a low-rank solver, extract the only solution (as num cols of Omega is assumed to be 1).
        if( isstruct(Z) )
            Z.left = Z.left{1};
            Z.right = Z.right{1};
        end
    
        % Assumes that Z is an n x ell matrix (i.e. that its low rank columns are not given in a factored form).
        if( config.verbosity > 0 )
            logprintf( 1, 'Done.\n----------------------------------------\n' );
            analyze_sylvester_solution( example, config, shift, Omega, Z, namedparams );
        end
    end
end


function analyze_sylvester_problem( example, config, shift, Omega, namedparams )
    n = example.size;
    nsqrt = sqrt(n);
    
    logprintf( 1, 'Example: ''%s''.\n', example.name );
    logprintf( 1, '\tEigenvalue problem / linear system size: %d\n', n );
    logprintf( 1, '\tSylvester equation size: %d x %d\n\n', nsqrt, nsqrt );

    % Just a sanity check for the generated matrices.
    if( config.verbosity > 1 )
        % Check if A and B have correct structure: B = kron(M, M), A = kron(M, K) + kron(K, M) + kron(VL, VR).
        logprintf( 2, '\tSanity check:\n');

        if( config.is_potential_separate == false )
            A_err = norm( example.A - (kron(example.M, example.K) + kron(example.K, example.M) + kron(example.VL, example.VR)), 'fro');
            B_err = norm( example.B - kron(example.M, example.M), 'fro');

            logprintf( 2, '\t\tnorm(A - (kron(M, K)+kron(K, M)+kron(VL, VR)) = %.2e, norm(B - kron(M, M)) = %.2e\n', A_err, B_err );
        end 

        % Check if A and B have correct structure: B = kron(M, M), A = kron(M, K) + kron(K, M) + sum_ell(kron(Vleft{ell}, Vright{ell})).
        A_sum = kron(example.M, example.K) + kron(example.K, example.M);
        for ell = 1:length(example.Vleft)
            A_sum = A_sum + kron(example.Vleft{ell}, example.Vright{ell});
        end
        A_err = norm( example.A - A_sum, 'fro');
        logprintf( 2, '\t\tnorm(A - (kron(M, K)+kron(K, M)+sum(kron(Vleft{ell}, Vright{ell})) = %.2e\n\n', A_err );
    end

    % Find where are the eigs of A and z*B-A.
    if( config.verbosity > 1 )
        if( nsqrt > 200 )
            logprintf( 2, '\tSkiping eigenvalues problem stats when nsqrt > 200.\n\n');
        else 
            eig_min = eigs(-example.A, 1, 'smallestreal');
            eig_max = eigs(-example.A, 1, 'largestreal');
            
            logprintf( 2, '\tEigenvalue problem stats:\n');
            logprintf( 2, '\t\tEigs of -A are in [%.2f, %.2f]\n', eig_min, eig_max );    
            logprintf( 2, '\t\tCountour: circle with center %.2f and radius %.2f; using shift z=%.2f%+.2fi.\n', ...
                example.center, example.radius, real(shift), imag(shift) );    

            eig_min = eigs(shift*example.B - example.A, 1, 'smallestreal');
            eig_max = eigs(shift*example.B - example.A, 1, 'largestreal');
            logprintf( 2, '\t\tReal parts of eigs of z*B-A are in [%.2f, %.2f]\n', real(eig_min), real(eig_max) );

            if( config.sylv_put_shift_in_lyap )
                % The z*MXM part goes to into the Lyapunov part, with K.
                eig_min = eigs(shift/2*example.M - example.K, 1, 'smallestreal');
                eig_max = eigs(shift/2*example.M - example.K, 1, 'largestreal');
                logprintf( 2, '\t\tReal parts of eigs of z/2*M-K are in [%.2f, %.2f]\n', real(eig_min), real(eig_max) );

                % Compute the spec. radius of the fixed point iteration matrix, depending on where we put the z*MXM part.
                % Let z*B-A = L + N, L = z*B - (kron(M, K) + kron(K, M)), N = -kron(VL, VR).
                % Then the iteration matrix is -L\N.
                L = shift*example.B - (kron(example.M, example.K) + kron(example.K, example.M));
                N = -kron(example.VL, example.VR);
            else 
                % The z*MXM part does not go into the Lyapunov part, but as a separate "N1*X*N2".
                eig_min = eigs(-example.K, 1, 'smallestreal');
                eig_max = eigs(-example.K, 1, 'largestreal');
                logprintf( 2, '\t\tReal parts of eigs of -K are in [%.2f, %.2f]\n', real(eig_min), real(eig_max) );

                % Compute the spec. radius of the fixed point iteration matrix, depending on where we put the z*MXM part.
                % Let z*B-A = L + N, L = -(kron(M, K) + kron(K, M)), N = -kron(VL, VR) + z*B.
                % Then the iteration matrix is -L\N.
                L = -(kron(example.M, example.K) + kron(example.K, example.M));
                N = -kron(example.VL, example.VR) + shift*example.B;
            end

            assert( norm((shift*example.B - example.A) - (L+N), 'fro') < 1e-6 );

            % Compute largest eigs of L^(-1)*N.
            LL = @(x) -L\(N*x);
            lam = eigs(LL, size(example.A, 1), 1, 'largestabs');
            fprintf( '\t\tSpec. radius of the fixed point iteration matrix L^(-1)*N = %.4f\n\n', abs(lam) );
        end
    end

    logprintf( 1, 'Right hand side has vectors of type: ''%s''\n', config.random_vectors_type );
    logprintf( 1, 'Number of vectors on the right hand side: %d\n\n', config.dim_right_projection );
end


function analyze_sylvester_solution( example, config, shift, Omega, Z, namedparams )
    n = example.size;
    nsqrt = sqrt(n);
    
    if( ~isstruct(Omega) )
        % Omega, Z are given as a matrix. 
        res = norm( (shift*example.B - example.A) * Z - Omega, 'fro' );
        logprintf( 1, '\tResidual of the solution: %.4e\n\n', res );
    else 
        % Omega, Z are given as a struct, check the F-norm of the Sylvester residual.
        % shift*MZM - (MZK + KZM + VR*Z*VL) = OM.r*OM.l'
        % These matrices are nsqrt x nsqrt in size:
        RHS = Omega.right * Omega.left';
        MZK = (example.M * Z.left) * (example.K' * Z.right)';
        KZM = (example.K * Z.left) * (example.M' * Z.right)';
        VZV = (example.VR * Z.left) * (example.VL' * Z.right)';
        MZM = (example.M * Z.left) * (example.M' * Z.right)';

        fprintf( 'Solution rank (number of cols): %d\n', size(Z.left, 2) );
        fprintf( 'Residual: %8.4e\n', norm( shift*MZM - (MZK + KZM + VZV) - RHS, 'fro' ) / norm( RHS, 'fro' ) );
        
        % Sanity check. Should be completely the same.
        X = Z.left*Z.right';
        fprintf( 'Residual unfolded: %.4e\n', norm( ( shift*example.B - example.A )*X(:) - RHS(:) ) / norm( RHS, 'fro' ) );
    end

    logprintf( 1, 'Ranks of columns in Sylvester solution (tol=1e-5):\n' );
    f = figure();
    for i = 1 : config.dim_right_projection
        if( ~isstruct(Omega) )
            s = svd( reshape(Z(:, i), nsqrt, nsqrt) );
        else 
            assert(config.dim_right_projection == 1);
            s = svd( reshape(X, nsqrt, nsqrt) );
        end

        ind = min( find( s ./ s(1) < 1e-5 ) );
        if( isempty(ind) )
            ind = nsqrt;
        end

        logprintf( 1, '\tColumn %02d -> sigma(%d)/sigma(1) = %.4e\n', i, ind, s(ind)/s(1) );

        % Plot the singular value decay.
        semilogy( s, 'DisplayName', sprintf( 'Column %d', i ) );
        hold on;
    end
    legend();
    title( sprintf('Singular values of the Sylvester solution, z = center + r*cis(%.2f)', namedparams.z_angle ) );
    hold off;
end
