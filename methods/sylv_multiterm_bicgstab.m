function [X1, X2, hist] = sylv_multiterm_bicgstab( A1, B1, A2, B2, C1, C2, N1, N2, config )
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Solve the generalized multiterm Sylvester equation of the form
%   A1*X*B1' + A2*X*B2' + N1{1}*X*N2{1}' + ... + N1{l}*X*N2{l}' + C1*C2' = 0
% using a variant of the preconditioned BiCGstab, Algorithm 2 from [1].
% 
% Preconditioners available:
%   M = I (no preconditioner)
%   M = use a few steps of ADI for A1*X*B1' + A2*X*B2'= P1*P2'.
%
% Input:
%   A1, A2, B1, B2 - structs with functions matvec, solve taking one parameter.
%   C1, C2 - matrices
%   N1, N2 - cell arrays containing matrices
%
% Output:
%   X1, X2 - low-rank factors of the solution X = X1*X2'.
%   hist - struct with entries niters (number of iterations taken), res (final residual).
%
% [1] Benner, Breiten - Low rank methods for a class of generalized Lyapunov equations and related issues (2013)

    % Convert A1, A2, B1, B2 to operators if they are not already.
    if( ~isstruct(A1) )
        A1 = mkop( A1 );
        A2 = mkop( A2 );
        B1 = mkop( B1 );
        B2 = mkop( B2 );
    end

    % Some initial setup.
    norm_C1C2 = norm(C1, 'fro') * norm(C2, 'fro');

    tol = config.biCGstab_tol;

    % Some hardcoded inner tolerances for the Sylvester ADI solver.
    tols_truncation = [1e-6, 1e-9, 1e-13, 1e-13, 1e-13];
    tols_adi = [1e-3, 1e-3, 1e-3, 1e-5, 1e-5];
    biCGstab_max_ranks = [40, 65, 85, 90, 90];

    config.max_adi_iters = 55;
    tol_truncation = tols_truncation(1);
    config.tol_adi = tols_adi(1);
    config.biCGstab_max_rank = biCGstab_max_ranks(1);

    maxiter = 100;
    n_N = length( N1 );

    % preconditioner = @preconditioner_eye;
    preconditioner = @preconditioner_ADI_one_by_one;

    logprintf( 3, '\tUsing tolerances: tol_trunc = %.2e, tol_adi = %.2e\n', tol_truncation, config.tol_adi );

    % Initial guess is X=0.
    X1 = []; X2 = [];
    
    % Initial residual, and the low-rank factorization of the first Sylvester eqn.
    R1 = C1; R2 = -C2;
    Rtilde1 = C1; Rtilde2 = -C2;
    rho = lr_dotprod( Rtilde1, Rtilde2, R1, R2 );
    P1 = R1; P2 = R2;
    [Ptilde1, Ptilde2] = preconditioner( A1, B1, A2, B2, N1, N2, P1, P2, config ); % Ptilde = M \ P
    [Ptilde1, Ptilde2] = low_rank_truncate( Ptilde1, Ptilde2, tol_truncation, config.biCGstab_max_rank );

    % Main loop.
    logprintf( 2, '\tSolving generalized Sylvester eqn. with %d x %d matrices\n', size(C1, 1), size(C2, 1) );
    norm_S = inf; norm_R = inf;
    for iter = 1 : maxiter
        logprintf( 2, '\tIteration %02d:\n', iter );

        % Tweak the inner tolerances depending on the iteration.
        tol_truncation = max(tols_truncation(min(iter, 4)) );
        config.tol_adi = tols_adi(min(iter, 4));
        config.biCGstab_max_rank = biCGstab_max_ranks(min(iter, 4));

        % Compute V = A(Ptilde).
        V1 = [A1.matvec(Ptilde1) A2.matvec(Ptilde1)];
        V2 = [B1.matvec(Ptilde2) B2.matvec(Ptilde2)];
        for i = 1 : n_N
            V1 = [V1 N1{i}*Ptilde1];
            V2 = [V2 N2{i}*Ptilde2];
        end

        % Compress V1, V2.
        old_size = size(V1, 2);
        [V1, V2] = low_rank_truncate( V1, V2, tol_truncation, config.biCGstab_max_rank );
        logprintf( 3, '\t\trank of V before truncation: %d, after: %d\n', old_size, size(V1, 2) );
        
        % Compute omega.
        om = lr_dotprod(Rtilde1, Rtilde2, R1, R2) / lr_dotprod(Rtilde1, Rtilde2, V1, V2);

        % Compute S, T in the factored form.
        S1 = [R1 -om*V1]; S2 = [R2 V2];
        [S1, S2, norm_S] = low_rank_truncate( S1, S2, tol_truncation, config.biCGstab_max_rank );

        % Stop if S is small enough.
        if( norm_S < tol * norm_C1C2 )
            X1 = [X1 om*Ptilde1]; X2 = [X2 Ptilde2];
            [X1, X2] = low_rank_truncate( X1, X2, tol_truncation, config.biCGstab_max_rank );
            break;
        end
        
        logprintf( 3, '\t\tprecond for S...rank(rhs)=%d\n', size(S1, 2) );
        [Stilde1, Stilde2] = preconditioner( A1, B1, A2, B2, N1, N2, S1, S2, config ); % Stilde = M \ S
        [Stilde1, Stilde2] = low_rank_truncate( Stilde1, Stilde2, tol_truncation, config.biCGstab_max_rank );
        logprintf( 3, '\t\t\trank of ADI Stilde after truncation: %d\n', size(Stilde1, 2) );

        % Compute T = A(Stilde).
        T1 = [A1.matvec(Stilde1) A2.matvec(Stilde1)];
        T2 = [B1.matvec(Stilde2) B2.matvec(Stilde2)];
        for i = 1 : n_N
            T1 = [T1 N1{i}*Stilde1];
            T2 = [T2 N2{i}*Stilde2];
        end

        % Compress T1, T2.
        old_size = size(T1, 2);
        [T1, T2] = low_rank_truncate( T1, T2, tol_truncation, config.biCGstab_max_rank );
        logprintf( 3, '\t\trank of V before truncation: %d, after: %d\n', old_size, size(T1, 2) );

        % Compute xi.
        xi = lr_dotprod( T1, T2, S1, S2 ) / lr_dotprod( T1, T2, T1, T2 );

        % Update X.
        X1 = [X1 om*Ptilde1 xi*Stilde1];
        X2 = [X2 Ptilde2 Stilde2];

        % Compress X1, X2.
        old_size = size(X1, 2);
        [X1, X2] = low_rank_truncate( X1, X2, tol_truncation, config.biCGstab_max_rank );
        logprintf( 3, '\t\trank of X before truncation: %d, after: %d\n', old_size, size(X1, 2) );

        % Compute the new residual.
        R1 = [-C1 -A1.matvec(X1) -A2.matvec(X1)];
        R2 = [ C2  B1.matvec(X2)  B2.matvec(X2)];
        for i = 1 : n_N
            R1 = [R1 -N1{i}*X1];
            R2 = [R2  N2{i}*X2];
        end

        % Compress R1, R2.
        old_size = size(R1, 2);
        [R1, R2, norm_R] = low_rank_truncate( R1, R2, tol_truncation, config.biCGstab_max_rank );
        logprintf( 3, '\t\trank of R before truncation: %d, after: %d\n', old_size, size(R1, 2) );

        % Stop if the residual is small.
        if( norm_R < tol * norm_C1C2 )
            break;
        end 

        % Compute rho, beta.
        old_rho = rho;
        rho  = lr_dotprod( Rtilde1, Rtilde2, R1, R2 );
        beta = rho/old_rho * om/xi;

        % Update P, Ptilde, V.
        P1 = [R1 beta*P1 -xi*beta*V1];
        P2 = [R2 P2 V2];

        % Compress P1, P2.
        old_size = size(P1, 2);
        [P1, P2] = low_rank_truncate( P1, P2, tol_truncation, config.biCGstab_max_rank );
        logprintf( 3, '\t\trank of P before truncation: %d, after: %d\n', old_size, size(P1, 2) );

        % Update Ptilde.
        logprintf( 3, '\t\tprecond for P...rank(rhs)=%d\n', size(P1, 2) );
        [Ptilde1, Ptilde2] = preconditioner( A1, B1, A2, B2, N1, N2, P1, P2, config ); % Ptilde = M \ P
        [Ptilde1, Ptilde2] = low_rank_truncate( Ptilde1, Ptilde2, tol_truncation, config.biCGstab_max_rank );
        logprintf( 4, '\t\t\trank of ADI Ptilde after truncation: %d\n', size(Ptilde1, 2) );

        logprintf( 2, '\t\t<strong>relnorm_S = %.4e, relnorm_R = %.4e</strong>\n', norm_S / norm_C1C2, norm_R / norm_C1C2 );
    end
  
    logprintf( 2, '\t\t<strong>relnorm_S = %.4e, relnorm_R = %.4e</strong>\n', norm_S / norm_C1C2, norm_R / norm_C1C2 );

    % The solution is X1*X2'.
    hist.niters = iter;
    hist.res = min(  norm_S / norm_C1C2, norm_R / norm_C1C2 );
end


function [Z1, Z2, norm_Z1Z2] = low_rank_truncate( Z1, Z2, tol, max_rank )
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

    r = min(r, max_rank);

    Z1 = Q1 * U(:, 1:r) * diag( sqrt(sigma(1:r)) );
    Z2 = Q2 * V(:, 1:r) * diag( sqrt(sigma(1:r)) );

    norm_Z1Z2 = sqrt(sum(sigma(1:r).^2));
end


function res = lr_dotprod( X1, X2, Y1, Y2 )
% Computes <X, Y> = trace(X'*Y) where X=X1*X2', Y=Y1*Y2'.
% trace(X2*X1'*Y1*Y2') = trace( (Y2'*X2) * (X1'*Y1) ).
    res = trace( (Y2'*X2) * (X1'*Y1) );
end


function [Ptilde1, Ptilde2] = preconditioner_eye( A1, B1, A2, B2, N1, N2, P1, P2, config )
    Ptilde1 = P1;
    Ptilde2 = P2;
end


function [Ptilde1, Ptilde2] = preconditioner_ADI_one_by_one( A1, B1, A2, B2, N1, N2, P1, P2, config )
    % A precoditioner to solving A(X) = P1*P2',
    % where A(X) = A1*X*B1' + A2*X*B2' + N1{1}*X*N2{1}' + ... + N1{l}*X*N2{l}'.
    % Here we run a few iterations of ADI, ignoring the N-part.
    % I.e., we use ADI for A1*X*B1' + A2*X*B2' = P1*P2'.
    % Columns of P1, P2 are grouped in blocks, and ADI is run for each group separately.
    % Column compression is done after each ADI solve.

    % Number of right-hand sides.
    r = size( P1, 2 );
    
    % Number of ADI shifts; hardcoded...
    l0A = 40;
    l0B = 40;

    % Arnoldi subspaces dimensions; hardcoded...
    kpA = 15;
    kmA = 25;
    kpB = 15;
    kmB = 25;
    
    % Shift generation. 
    % Build Arnoldi subspace with E \ A.
    [Hp, Vp] = arn_pl( A2.X, A1.X, kpA, P1*ones(r,1) );
    rwp = eig(Hp(1:kpA,1:kpA));
      
    % Build Arnoldi subspace with inv( E \ A ).
    [Hm, Vm] = arn_inv( A2.X, A1.X, kmA, P1*ones(r,1) );
    rwm = ones(kmA,1) ./ eig(Hm(1:kmA,1:kmA));
    sA = sort( [rwp; rwm] );
    sA = sA( 1:l0A );
     
    % Build Arnoldi subspace with C \ B.
    [Hp, Vp] = arn_pl( B1.X, B2.X, kpB, P2*ones(r,1) );
    rwp = eig(Hp(1:kpB,1:kpB));
      
    % Build Arnoldi subspace with inv( C \ B ).
    [Hm, Vm] = arn_inv( B1.X, B2.X, kmB, P2*ones(r,1) );
    rwm = ones(kmB,1)./eig( Hm(1:kmB,1:kmB) );
    sB=sort( [rwp; rwm] );
    sB = sB( 1:l0B );
    
    % Compute pseudo minmax.
    [sA, sB] = pseudominmax(sA, sB);

    n_ADI_iterations = config.max_adi_iters;
    tol = config.tol_adi;
    
    % Solve the (generalized) Sylvester equation using ADI with the generated shifts.
    % function [Z,D,Y,res,niter,timings]=lr_gfadi_diss(A,B,E,C,F,G,pA,pB,maxiter,rtol,tcrit)
    %   X=Z*D*Y' 
    %   A*X*C - E*X*B = F*G'      
    % [Z, D, Y, res, niter, timings] = lr_gfadi_diss( A, -B, E, C, F, -G, sA, -sB, n_ADI_iterations, tol );
    Ptilde1 = []; Ptilde2 = [];
    i_step = 10; % Process columns of the RHS in groups of 10.
    for i = 1 : i_step : r 
        [Z, D, Y, res, n_iters, timings] = lr_gfadi_cplx( A1.X, -B2.X', A2.X, B1.X', P1(:, i:min(i+i_step-1, r)), P2(:, i:min(i+i_step-1, r)), sA, -sB, n_ADI_iterations, tol );
        Ptilde1 = [Ptilde1 Z*D];
        Ptilde2 = [Ptilde2 Y];

        % Hardcoded trucation tolerance...
        [Ptilde1, Ptilde2, ~] = low_rank_truncate( Ptilde1, Ptilde2, 1e-9, config.biCGstab_max_rank );
    end
end


% Convert a matrix to an operator.
function A = mkop( ZZ )
    A.X = ZZ;
    % [A.L, A.U] = lu(A.X);
    A.matvec = @(X) (ZZ * X);
    A.solve  = @(X) (ZZ \ X);
end
