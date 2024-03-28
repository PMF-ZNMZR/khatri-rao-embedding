function [X, lam,res_vec,rank_vec,abs_error] = lrlobpcg_blk_folded( example, config, shift, ref_sol)
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
% 
% Implements the low-rank LOBPCG algorithm from [1] with block low-rank
% format. Suppose we have X=[x_1,x_2,...,] where x_i is a vector. We then store
% matrices X1,X3, and a 3d tensor X2 such that vec^{-1}{x_i}=X1*X2(:,:,i)*X3'.
%
% In this fucntion we apply A to block low rank vector twice (thus, "folded").
%
% [1] Kressner, Tobler - Preconditioned Low-Rank Methods for High-Dimensional Elliptic PDE Eigenvalue Problems (2011)
    res_vec=[];
    rank_vec=[];
    abs_error=[];
    n = example.size;
    nsqrt = sqrt( n );

    assert( norm( example.M - speye(nsqrt), 'fro' ) == 0 );

    % Tolerances.
    tol = config.tol;         % Stop when eigenpair residual is < tol. In the plot, we only stop the iteration by maxiter (see below).
    tol_truncation = 1e-7;    % Low-rank recompression of the columns during the process.
    maximal_rank=120;

    config.tol_adi = 1e-15;    % Stop the ADI preconditioner when this rel. tolerance is reached.
    config.max_adi_iters = 12; % Maximum number of ADI iterations.

    logprintf( 1, '\tStopping tolerance for the eigenpair residual: %.2e\n', tol );
    logprintf( 1, '\tTruncation threshold for low-rank compression: %.2e\n', tol_truncation );
    logprintf( 1, '\tADI preconditioner:\n\t\ttolerance: %.2e\n\t\tmax iterations: %d\n\n', config.tol_adi, config.max_adi_iters );

    % Maximum number of LOBPCG iterations.
    maxiter = 450;

    % Block size 
    blk_size=3;               % Block size of vectors.
    target_no=1;              % Tagret number of eigenpairs.

    % Select the preconditioner.
    preconditioner = @preconditioner_ADI_one_by_one;
   
    example.shift = shift; %shift for ensure psd.

    % Generate the random matrix Omega
    rnd_state = rng(); rng(12345); 
    Omega = generate_Omega( n, blk_size, 'rank-one' );
    rng(rnd_state);
    [X_L, R_L] = qr(Omega.left, "econ");
    [X_R, R_R] = qr(Omega.right, "econ");
    X_core=zeros(blk_size,blk_size,blk_size);
    for i=1:blk_size
        X_core(i,i,i)=1;
        X_core(:,:,i)=R_L*X_core(:,:,i)*R_R';
    end
   
    % orthonormalize X
    [L_X,flag]=chol(Gram_matrix(X_L,X_R,X_core,X_L,X_R,X_core,blk_size));

    %%update X2 i.e. X2_i*L_X
    L_X=real(L_X);
    L_X_inv=eye(blk_size)/(L_X);        
    temp_X_core=zeros(size(X_core));
    for count=1:blk_size
        temp_X_core(:,:,count)=ttm(tensor(X_core), L_X_inv(:,count)', 3);
    end 
    X_core=temp_X_core;
    clear temp_X_core
    
    %calculate AX
    [AX_L, AX_R,AX_core] = applyA( example, X_L, X_R,X_core );    

    %initial eigenpairs approximation 
    [Y, lam_tilde] = eig( Gram_matrix(X_L,X_R,X_core,AX_L,AX_R,AX_core,blk_size), eye(blk_size) ,"chol");
    [lam, idx] = mink(diag(lam_tilde),blk_size);
    y = Y(:, idx);
        
    rank_vec=[rank_vec,size(X_L, 2)];
    
    temp_X_core=zeros(size(X_core));
    for count=1:blk_size
        temp_X_core(:,:,count)=ttm(tensor(X_core), y(:,count)', 3);
    end 
    X_core=temp_X_core;
    clear temp_X_core
     [AX_L, AX_R,AX_core] = applyA( example, X_L, X_R,X_core );  

    P1 = tensor(zeros( nsqrt, 0 ));
    P2 = tensor(zeros( nsqrt, 0 ,blk_size));

    % main loop
    for iter = 1:maxiter
        logprintf( 2, '\tIteration %02d:\n', iter );

        % Compute the residual.
        R_L = [AX_L X_L];
        R_R = [AX_R X_R];
        R_core=zeros(size(AX_core,1)+size(X_core,1),size(AX_core,2)+size(X_core,2),blk_size);
        for i=1:blk_size
            R_core(:,:,i)=blkdiag(AX_core(:,:,i),-lam(i,1)*X_core(:,:,i));
        end
        
        res = sqrt( diag( Gram_matrix(R_L,R_R,R_core,R_L,R_R,R_core,blk_size) ) ).';

        res_vec=[res_vec;res];
        abs_error=[abs_error;abs((sqrt(ref_sol)-0.2)-(sqrt(lam)-0.2)+example.shift)'];
        
        % It stops if it reach the number of eigenpairs required such that residual < tol.
        converged_no=sum(res(:)<=tol); % no. of converged eigenpairs.
        if( converged_no >= target_no )
            logprintf( 1, '\t\t<strong>Residual = %.2e</strong>\n', res );
            break;
        end

        % Apply the preconditioner 
        [R_L, R_R,R_core] = low_rank_truncate_blk( R_L, R_R,R_core, tol_truncation ,maximal_rank);
        temp=zeros(size(R_R,1),size(R_L,2),blk_size);
        for i=1:blk_size
            temp(:,:,i)=R_R*R_core(:,:,i)';
        end
        temp=tenmat(temp,2)';

        [R1, R2] = preconditioner( example, config, R_L, temp.data ,blk_size); % R = B \ R and return R in mode 1 low-rank format
                
        % cast to Block low-rank format
        temp=reshape(permute(reshape(reshape(R2',[],1),size(R2,2),nsqrt,blk_size),[2 1 3]),nsqrt,blk_size*size(R2,2));
        [Q_temp,R_temp]=qr(temp,'econ');
        C=permute(reshape(R_temp,size(R_temp,1),size(R2,2),blk_size),[2 1 3]);
                
        [R_L, R_R,R_core] = low_rank_truncate_blk( R1,Q_temp,C, tol_truncation ,maximal_rank);

        % orthonormalize R
        [L_R,flag]=chol(Gram_matrix(R_L,R_R,R_core,R_L,R_R,R_core,blk_size)); 
        L_R=real(L_R);
        L_R_inv=eye(blk_size)/(L_R);
        temp_R_core=zeros(size(R_core));
        for count=1:blk_size
            temp_R_core(:,:,count)=ttm(tensor(R_core), L_R_inv(:,count)', 3);
        end 
        R_core=temp_R_core;
        clear temp_R_core
        
        %calculate AR
        [AR_L, AR_R,AR_core] = applyA( example, R_L, R_R,R_core );
        
        %orthonormalize P and build Mtilde and Atilde
        if iter==1
            Atilde=[Gram_matrix(X_L,X_R,X_core,AX_L,AX_R,AX_core,blk_size),Gram_matrix(X_L,X_R,X_core,AR_L,AR_R,AR_core,blk_size);
                Gram_matrix(R_L,R_R,R_core,AX_L,AX_R,AX_core,blk_size),Gram_matrix(R_L,R_R,R_core,AR_L,AR_R,AR_core,blk_size)];
                          
            Mtilde=[Gram_matrix(X_L,X_R,X_core,X_L,X_R,X_core,blk_size),Gram_matrix(X_L,X_R,X_core,R_L,R_R,R_core,blk_size);
                Gram_matrix(R_L,R_R,R_core,X_L,X_R,X_core,blk_size),Gram_matrix(R_L,R_R,R_core,R_L,R_R,R_core,blk_size);];
        else
            % orthonormalize R
            [L_P,flag]=chol(Gram_matrix(P_L,P_R,P_core,P_L,P_R,P_core,blk_size));
            
            %%update 
            L_P=real(L_P);
            L_P_inv=eye(blk_size)/(L_P);        
            temp_P_core=zeros(size(P_core));
            for count=1:blk_size
                temp_P_core(:,:,count)=ttm(tensor(P_core), L_P_inv(:,count)', 3);
            end 
            P_core=temp_P_core;
            clear temp_P_core
               
            [AP_L, AP_R,AP_core] = applyA( example, P_L, P_R,P_core ); 

            Atilde=[Gram_matrix(X_L,X_R,X_core,AX_L,AX_R,AX_core,blk_size),Gram_matrix(X_L,X_R,X_core,AR_L,AR_R,AR_core,blk_size),Gram_matrix(X_L,X_R,X_core,AP_L,AP_R,AP_core,blk_size);
                Gram_matrix(R_L,R_R,R_core,AX_L,AX_R,AX_core,blk_size),Gram_matrix(R_L,R_R,R_core,AR_L,AR_R,AR_core,blk_size),Gram_matrix(R_L,R_R,R_core,AP_L,AP_R,AP_core,blk_size);
                Gram_matrix(P_L,P_R,P_core,AX_L,AX_R,AX_core,blk_size),Gram_matrix(P_L,P_R,P_core,AR_L,AR_R,AR_core,blk_size),Gram_matrix(P_L,P_R,P_core,AP_L,AP_R,AP_core,blk_size);];
           
            Mtilde=[Gram_matrix(X_L,X_R,X_core,X_L,X_R,X_core,blk_size),Gram_matrix(X_L,X_R,X_core,R_L,R_R,R_core,blk_size),Gram_matrix(X_L,X_R,X_core,P_L,P_R,P_core,blk_size);
                Gram_matrix(R_L,R_R,R_core,X_L,X_R,X_core,blk_size),Gram_matrix(R_L,R_R,R_core,R_L,R_R,R_core,blk_size),Gram_matrix(R_L,R_R,R_core,P_L,P_R,P_core,blk_size);
                Gram_matrix(P_L,P_R,P_core,X_L,X_R,X_core,blk_size),Gram_matrix(P_L,P_R,P_core,R_L,R_R,R_core,blk_size),Gram_matrix(P_L,P_R,P_core,P_L,P_R,P_core,blk_size);];

        end 
        Atilde=(Atilde'+Atilde)./2;
        Mtilde=(Mtilde'+Mtilde)./2;
         
        % Compute the smallest eigenpair of the (3*blk_size)x(3*blk_size) pencil (Atilde, Mtilde).
        [Y, lam_tilde] = eig( Atilde, Mtilde ,"chol");
        [lam, idx] = mink(diag(lam_tilde),blk_size);
        y = Y(:, idx);

        % Update P, X.  
        if iter==1
            P_L=R_L;
            P_R=R_R;
            P_core=zeros(size(R_core));
            for count=1:blk_size
               P_core(:,:,count)=ttm(tensor(R_core), y(blk_size+1:2*blk_size,count)', 3);
            end
        else
           temp_R_core=zeros(size(R_core));
           temp_P_core=zeros(size(P_core));
           
           for count=1:blk_size
                temp_R_core(:,:,count)=ttm(tensor(R_core), y(blk_size+1:2*blk_size,count)', 3);
                temp_P_core(:,:,count)=ttm(tensor(P_core), y(2*blk_size+1:3*blk_size,count)', 3);    
           end

           P_L = [R_L P_L]; 
           P_R = [R_R P_R]; 
           P_core=zeros(size(R_core,1)+size(P_core,1),size(R_core,2)+size(P_core,2),blk_size);
           for i=1:blk_size
               P_core(:,:,i)=blkdiag(temp_R_core(:,:,i),temp_P_core(:,:,i));
           end
        end
        
        clear temp_R2 temp_p2
        
        old_size = size(P_L, 2);
        [P_L, P_R, P_core] = low_rank_truncate_blk( P_L, P_R,P_core, tol_truncation ,maximal_rank);
        
        %update X
        X_L = [X_L P_L]; 
        X_R = [X_R P_R]; 
        temp_X_core=zeros(size(X_core));
        for count=1:blk_size
            temp_X_core(:,:,count)=ttm(tensor(X_core), y(1:blk_size,count)', 3);
        end    
  
        X_core=zeros(size(X_core,1)+size(P_core,1),size(X_core,2)+size(P_core,2),blk_size);

        for i=1:blk_size
            X_core(:,:,i)=blkdiag(temp_X_core(:,:,i),P_core(:,:,i));
        end

        old_size = size(X_L, 2);
        [X_L, X_R,X_core] = low_rank_truncate_blk( X_L, X_R,X_core, tol_truncation ,maximal_rank);
        logprintf( 3, '\t\trank of X before truncation: %d, after: %d\n', old_size, max(size(X_core, 1),size(X_core, 2)) );
        rank_vec=[rank_vec,size(X_L, 2)];
        
        [AX_L, AX_R,AX_core] = applyA( example, X_L, X_R,X_core );

        logprintf( 1, '\t\t<strong>Residual = %.2e</strong>\n', res );
    end

    logprintf( 1, '\tComputed eigenvector has rank (num. cols. in lr factor): %d\n', size(X_L, 2) );

    X = block_Tensor_to_vec( X_L, X_R,X_core,blk_size );
    lam = lam - example.shift;
end


function [AX1, AX2,AX_core] = applyA( example, X1, X2,X_core )
    % Compute A*A*X where A = kron(M, K) + kron(K, M) + kron(VR, VL) + shift*I.
 
    nsqrt = length(example.K);

    % Keep the symmetry in shift.
    AX1 = [(example.K + example.shift/2*speye(nsqrt))*X1, example.M*X1, example.VR*X1];
    AX2 = [example.M*X2, (example.K + example.shift/2*speye(nsqrt))*X2, example.VL*X2];
    %AX2 = cat(2,ttm( X2,example.M, 1).data, ttm( X2,(example.K + example.shift/2*speye(nsqrt)), 1).data, ttm( X2,example.VL, 1).data);
    AX_core=zeros(3*size(X_core,1),3*size(X_core,2),size(X_core,3));
    for i=1:size(X_core,3)
        AX_core(:,:,i)=blkdiag(X_core(:,:,i),X_core(:,:,i),X_core(:,:,i));
    end

    % apply A second time.
    AX1 = [(example.K + example.shift/2*speye(nsqrt))*AX1, example.M*AX1, example.VR*AX1];
    AX2 = [example.M*AX2, (example.K + example.shift/2*speye(nsqrt))*AX2, example.VL*AX2];
    %AX2 = cat(2,ttm( X2,example.M, 1).data, ttm( X2,(example.K + example.shift/2*speye(nsqrt)), 1).data, ttm( X2,example.VL, 1).data);
    AX2_core=zeros(3*size(AX_core,1),3*size(AX_core,2),size(AX_core,3));
    for i=1:size(X_core,3)
        AX2_core(:,:,i)=blkdiag(AX_core(:,:,i),AX_core(:,:,i),AX_core(:,:,i));
    end
    AX_core=AX2_core;
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
    
    [U, S, V] = svd( R1R2t );
    sigma = [diag(S); 0];

    nrmF = sqrt( sum(sigma.^2) );
    for r = 1:length(sigma)-1
        if( sqrt(sum(sigma(r+1:end).^2)) < tol * nrmF )
            break;
        end
    end

    Z1 = Q1 * U(:, 1:r) * sqrt(S(1:r, 1:r));
    Z2 = Q2 * V(:, 1:r) * sqrt(S(1:r, 1:r));

    norm_Z1Z2 = sqrt(sum(sigma(1:r).^2));
end


function [U] = low_rank_truncate_without_qr( Z1, tol ,maximal_rank)
    % Obtain left factors of Z1
    % returns matrices U such that
    %   || U'UZ1_Z1 ||_F <= tol./sqrt(2)*|| Z1||_F,
    % where the rank of Z1 is the smallest integer such that the above inequality holds. 
    [U, S, V] = svd( Z1,"econ");
    sigma = [diag(S); 0];

    nrmF = sqrt( sum(sigma.^2) );
    for r = 1:length(sigma)-1
        if( sqrt(sum(sigma(r+1:end).^2)) < tol./sqrt(2) * nrmF || r==maximal_rank )
            break;
        end
    end

    U = U(:,1:r);
end


function [Z1, Z2,C2] = low_rank_truncate_blk( U1, U2,C, tol ,maximal_rank)
    % trucate block low rank tensor 

    [U1,r1]=qr(U1,'econ');
    [U2,r2]=qr(U2,'econ');
    temp=zeros(size(r1,1),size(r2,1),size(C,3));

    for i=1:size(C,3)
        temp(:,:,i)=r1*C(:,:,i)*r2';
    end
    C=temp;

    C1=reshape( C, [size(C,1), size(C,2)*size(C,3)] );
    [Z1] = low_rank_truncate_without_qr( C1, tol ,maximal_rank);

    C1=permute(C,[2 1 3]);
    C1=reshape( C1, [size(C1,1), size(C1,2)*size(C1,3)] );
    [Z2] = low_rank_truncate_without_qr(  C1, tol ,maximal_rank);

    C2=ttm(ttm(tensor(C), Z1',1),Z2',2);
    C2=C2.data;
    Z1=U1*Z1;
    Z2=U2*Z2;
end


function X_vec = block_Tensor_to_vec( X1, X2,X_core,blk_size )
    temp=zeros(size(X2,1),size(X1,2),blk_size);
    for i=1:blk_size
        temp(:,:,i)=X2*X_core(:,:,i)';
    end

    X_vec=permute(temp,[2 1 3]);
    X_vec=reshape( X_vec, [size(X_vec,1), size(X_vec,2)*size(X_vec,3)] );
    X_vec=X1*X_vec;
    X_vec=reshape(X_vec,[],blk_size);
end


function Y = Gram_matrix(XL,XR,X_core,ZL,ZR,Z_core,blk_size)
    % calculate Gram matrix by trace. 
    XZL=XL'*ZL;
    XZR=ZR'*XR;
    Y=zeros(blk_size,blk_size);
    for i=1:blk_size
        for j=1:blk_size
            Y(i,j)=trace(X_core(:,:,i)'*XZL*Z_core(:,:,j)*XZR);
        end
    end
end


function [Ptilde1, Ptilde2] = preconditioner_ADI_one_by_one( example, config, P1, P2 ,blk_size)
    % A precoditioner to solving A(X) = R1*R2',
    % where A = kron(M, K) + kron(K, M) + kron(VR, VL) with M = I.
    % Here we run 5 iterations of ADI, ignoring the V-part (!)
    % I.e., we use ADI for M*X*K' + K*X*M' = P1*P2'.
    % ADI is run for each column of P1, P2 separately, and column compression is done after each ADI solve.

    % Number of right-hand sides.
    r = size( P1, 2 );
    r2 = size( P2, 1 );
    
    % number of ADI shifts
    l0A = 40;
    l0B = 40;

    % Arnoldi subspaces dimensions.
    kpA = 15;
    kmA = 25;
    kpB = 15;
    kmB = 25;
    
    nsqrt = length( example.K );
    A1.X = example.M;
    A2.X = (example.K)^2 + example.shift/2*speye(nsqrt);
    B1.X = kron(eye(blk_size), A2.X);
    B2.X = kron(eye(blk_size), A1.X);

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
    i_step = 10;
    for i = 1 : i_step : r 
        [Z, D, Y, res, n_iters, timings] = lr_gfadi_cplx( A1.X, -B2.X', A2.X, B1.X', P1(:, i:min(i+i_step-1, r)), P2(:, i:min(i+i_step-1, r)), sA, -sB, n_ADI_iterations, tol );
        Ptilde1 = [Ptilde1 Z*D];
        Ptilde2 = [Ptilde2 Y];

        [Ptilde1, Ptilde2, ~] = low_rank_truncate( Ptilde1, Ptilde2, 1e-5 );
    end

    % Check the norm, naively.
    if( config.verbosity >= 4 )
        % logprintf( 4, '\t\t\tADI: n_iters = %d, rank = %d, rel. res. norm = %.4e [reported: %.4e]\n', ...
        %     n_iters, ...
        %     size(Ptilde1, 2), ...
        %     norm( A1.X*Ptilde1*(B1.X*Ptilde2)' + A2.X*Ptilde1*(B2.X*Ptilde2)' - P1*P2', 'fro' ) / norm(P1, 'fro') / norm(P2, 'fro'), res(1, end) );
        logprintf( 4, '\t\t\tADI: n_iters = %d, rank = %d, rel. res. norm = %.4e\n', ...
            n_iters, ...
            size(Ptilde1, 2), ...
            res(1, end) );
    end
end
