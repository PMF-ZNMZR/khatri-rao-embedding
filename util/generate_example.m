function example = generate_example( example_name, options )
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Generates test matrices for numerical experiments.
    arguments 
        example_name;
        options.n = 100;              % Default problem size.
        options.output_mode = 'both'; % Default output mode of the function, see below.

        % If is_potential_separate=true, then K is always the discrete Laplacian. If false, then parts 
        % of the potential are incorporated into the K matrix.
        options.is_potential_separate = false; 
    end
%
% Consider the pencil (A, B) resulting from discretization of
%   -laplace(u) + potential*u = lam*u.
%
% For some potentials, A and B have the following form (*):
%   A = kron(M, K) + kron(K, M) + kron(VL, VR)
%   B = kron(M, M).
% When using finite differences, M = I.
%
% For the given example_name, this function returns a struct with the following fields:
%   size = n^2 = length(A)
%   name = name of the example
%   center = center of the contour (circle) for integration
%   radius = radius of the contour (circle) for integration
%   num_eigs = number of the eigenvalues that are inside the contour (precomputed)
%   is_symmetric = true if the problem is symmetric (i.e. the eigenvalues are real)
%   V = potential evaluated at each point of the domain (n x n matrix)
%
% If output_mode == 'factored' or output_mode == 'both', then the following fields are also computed:
%   M = mass matrix
%   K = stiffness matrix
%   VL, VR = factored potential matrices
%   Vleft{1}, ..., Vleft{ell}, Vright{1}, ..., Vright{ell} = factored potential matrices (duplicates of VL, VR when ell = 1)
%       Potential is kron(Vleft{1}, Vright{1}) + ... + kron(Vleft{ell}, Vright{ell}).
%
% If options.is_potential_separate == true, then:
%   K is always discrete Laplacian;
% otherwise,
%   K may take part of the potential, and a warning is displayed.
%
% If output_mode == 'full' or output_mode == 'both', then the following fields are also computed:
%   A
%   B
% 
    switch( example_name )
        case 'sum-of-squares'
            example = get_example_sum_of_squares(options);

        case 'sum-of-squares-minus-xy'
            example = get_example_sum_of_squares_minus_xy(options);

        case 'deal-ii'
            example = get_example_deal_ii(options);

        case 'trefethen-exp'
            example = get_example_trefethen_exp(options);

        case 'double-well'
            example = get_example_double_well(options);

        case 'singular-one-over-sum-squares'
            example = get_example_singular_one_over_sum_squares(options);

        case 'gauss'
            example = get_example_gauss(options);

        case 'gauss-easy'
            example = get_example_gauss_easy(options);

        case 'discontinuous'
            example = get_example_discontinuous(options);

        case 'matieu'
            example = get_example_matieu(options);

        case 'matieu_shifted'
            example = get_example_matieu_shifted(options);

        otherwise
            error( 'get_example :: unknown example_name' );
    end
end


function example = get_example_sum_of_squares(options)
    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    example = struct();
    example.name = 'sum-of-squares';
    n = options.n;

    a = -1;
    b = 1;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    
    k = 1;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);

    X = spdiags(xx(:), 0, n^2, n^2); 
    Y = spdiags(yy(:), 0, n^2, n^2); 
    V = k*(X.^2+Y.^2);

    example.V = reshape( diag(V), n, n );

    if( output_factored )
        if( options.is_potential_separate )
            example.M = speye( n );
            example.K = -T;

            % Cannot put the potentital only into VL, VR.
            % example.VL = sparse( n, n ); 
            % example.VR = sparse( n, n ); 

            example.Vleft{1} = speye( n );
            example.Vright{1} = k*spdiags( (x').^2, 0, n, n); 

            example.Vleft{2} = k*spdiags( (x').^2, 0, n, n); 
            example.Vright{2} = speye( n );
        else
            % warning( 'Part of the potential is in the K matrix.' );
            example.M = speye( n );
            example.K = -T + k*spdiags( (x').^2, 0, n, n);
            example.VL = sparse( n, n ); 
            example.VR = sparse( n, n ); 

            example.Vleft{1} = example.VL; 
            example.Vright{1} = example.VR; 
        end
    end

    if( output_full )
        A = kron( speye(n), T ) + kron( T, speye(n) );
        
        example.A = -A + V;
        example.B = speye( n^2 );
    end

    % [V,D] = eigs(example.A,5,'smallestreal');
    % diag(D)
    % pause
    % 5.1937e+00
    % 1.2747e+01
    % 1.2747e+01
    % 2.0299e+01
    % 2.5107e+01
    example.center = 1.2747e+01;
    example.radius = 0.9000e+01;
    
    example.num_eigs = 4;
    example.is_symmetric = true;

    example.size = n^2;
end


function example = get_example_sum_of_squares_minus_xy(options)
    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    example = struct();
    example.name = 'sum-of-squares-minus-xy';
    n = options.n;
    
    a = -1;
    b = 1;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    
    k = 1;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);
    
    X=spdiags(xx(:), 0, n^2, n^2); 
    Y=spdiags(yy(:), 0, n^2, n^2); 
    V = 1/2 * (X.^2 + Y.^2 - X.*Y);
    example.V = reshape( diag(V), n, n );

    if( output_factored )
        if( options.is_potential_separate )
            example.M = speye( n );
            example.K = -T;

            % Cannot put the potentital only into VL, VR.
            % example.VL = sparse( n, n ); 
            % example.VR = sparse( n, n ); 

            example.Vleft{1} = speye( n );
            example.Vright{1} = 1/2*spdiags( (x').^2, 0, n, n); 

            example.Vleft{2} = 1/2*spdiags( (y').^2, 0, n, n); 
            example.Vright{2} = speye( n );

            example.Vleft{3} = sqrt(1/2)*spdiags( x', 0, n, n);
            example.Vright{3} = -sqrt(1/2)*spdiags( y', 0, n, n);
        else
            % warning( 'Part of the potential is in the K matrix.' );

            example.M = speye( n );
            example.K = -T + 1/2*spdiags( (x').^2, 0, n, n);
            example.VL = sqrt(1/2)*spdiags( x', 0, n, n);
            example.VR = -sqrt(1/2)*spdiags( y', 0, n, n);

            example.Vleft{1} = example.VL;
            example.Vright{1} = example.VR; 
        end
    end

    if( output_full )
        A = kron( speye(n), T ) + kron( T, speye(n) );
        example.A = -A + V;
        example.B = speye( n^2 );
    end

    % [V,D] = eigs(example.A,5,'smallestreal');
    % diag(D)
    % pause
    % 5.0644e+00
    % 1.2477e+01
    % 1.2606e+01
    % 2.0017e+01
    % 2.4887e+01
    example.center = 1.2606e+01;
    example.radius = 0.9000e+01;
    
    example.num_eigs = 4;
    example.is_symmetric = true;    

    example.size = n^2;
end


function example = get_example_deal_ii(options)
% Potential from Bangerth's deal.II library with a = -1, b = 1; -> leads to
% relatively high ranks and indefinite A! Seems to match eigenvalues stated
% on https://www.dealii.org/developer/doxygen/deal.II/step_36.html

    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    assert( ~output_factored, 'Example deal-ii cannot be generated in factored form. Call generate_example with option output_mode=''full''.' );

    example = struct();
    example.name = 'deal-ii';
    n = options.n;

    a = -1;
    b = 1;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    A = kron( speye(n), T ) + kron( T, speye(n) );
    
    k = 1;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);
    
    X=spdiags(xx(:), 0, n^2, n^2); 
    Y=spdiags(yy(:), 0, n^2, n^2); 

    V = -100 * ( sqrt(xx.^2+yy.^2)<3/4 ).*(xx.*yy>0) + (-5)*( sqrt(xx.^2+yy.^2)<3/4 ).*(xx.*yy<=0);
    example.V = V;
    V=spdiags(V(:), 0, n^2, n^2); 
    
    example.A = -A + V;
    example.B = speye( n^2 );

    % [V,D] = eigs(example.A,5,'smallestreal');
    % diag(D)
    % pause
    % -7.4359e+01
    % -7.2842e+01
    % -4.3007e+01
    % -4.2501e+01
    % -3.7364e+01
    example.center = -5.8430e+01;
    example.radius =  1.8000e+01;
    
    example.num_eigs = 4;
    example.is_symmetric = true;

    example.size = n^2;
end


function example = get_example_trefethen_exp(options)
% Potential from Trefethen's book with a = -1, b = 1;
    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    example = struct();
    example.name = 'trefethen-exp';
    n = options.n;

    a = -1;
    b = 1;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    
    k = 1;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);
    
    X=spdiags(xx(:), 0, n^2, n^2); 
    Y=spdiags(yy(:), 0, n^2, n^2); 
    
    V = exp(10*(yy-xx-1));
    example.V = V;
    V = spdiags(V(:), 0, n^2, n^2); 
    
    if( output_full )
        A = kron( speye(n), T ) + kron( T, speye(n) );
        example.A = -A + V;
        example.B = speye( n^2 );
    end

    if( output_factored )
        example.M = speye( n );
        example.K = -T;
        example.VL = exp(-5)*spdiags( exp(-10*x'), 0, n, n);
        example.VR = exp(-5)*spdiags( exp(10*y'), 0, n, n);

        example.Vleft{1} = example.VL;
        example.Vright{1} = example.VR;
    end

    % [V,D] = eigs(example.A,5,'smallestreal');
    % diag(D)
    % pause
    % 5.0845e+00
    % 1.2361e+01
    % 1.3085e+01
    % 2.0691e+01
    % 2.4860e+01    
    example.center = 1.2888e+01;
    example.radius = 0.9000e+01;
    
    example.num_eigs = 4;
    example.is_symmetric = true;

    example.size = n^2;
end


function example = get_example_gauss(options)
% Gauss potential with a = -5, b = 5; and depth 50.

    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    example = struct();
    example.name = 'gauss';
    n = options.n;
    
    a = -5;
    b = 5;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);
    
    V = 50*exp(-xx.^2-yy.^2);
    
    example.V = V;
    
    V = spdiags(V(:), 0, n^2, n^2); 
    
    if( output_full )
        A = kron( speye(n), T ) + kron( T, speye(n) );
        example.A = -A - V;
        example.B = speye( n^2 );
    end 

    if( output_factored )
        example.M = speye( n );
        example.K = -T;
        example.VL = sqrt(50)*spdiags( exp(-(x').^2), 0, n, n);
        example.VR = -sqrt(50)*spdiags( exp(-(y').^2), 0, n, n);

        example.Vleft{1} = example.VL;
        example.Vright{1} = example.VR;
    end

    % [V,D] = eigs(example.A,7,'smallestreal');
    % diag(D)
    % pause
    % -36.9231
    % -24.9776
    % -24.9776
    % -15.3439
    % -14.3033
    % -14.2416
    example.center = -40;
    example.radius = 30;
    
    example.num_eigs = 6; % 10
    example.is_symmetric = true;

    example.size = n^2;
end


function example = get_example_gauss_easy(options)
% Gauss potential with a = -5, b = 5; and depth 20.  
    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    example = struct();
    example.name = 'gauss-easy';
    % n = 100; k = 20;
    n = options.n;
    
    a = -5;
    b = 5;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);
    
    V = exp(-xx.^2-yy.^2);
    
    example.V = V;
    
    V = spdiags(V(:), 0, n^2, n^2); 
    
    if( output_full )
        A = kron( speye(n), T ) + kron( T, speye(n) );
        example.A = -A - V;
        example.B = speye( n^2 );
    end 

    if( output_factored )
        example.M = speye( n );
        example.K = -T;
        % example.VL = sqrt(20)*spdiags( exp(-(x').^2), 0, n, n);
        % example.VR = -sqrt(20)*spdiags( exp(-(y').^2), 0, n, n);
        example.VL = spdiags( exp(-(x').^2), 0, n, n);
        example.VR = -spdiags( exp(-(y').^2), 0, n, n);

        example.Vleft{1} = example.VL;
        example.Vright{1} = example.VR;
    end

    % [V,D] = eigs(example.A,7,'smallestreal');
    % diag(D)
    % pause
    % 0.0421
    % 0.4714
    % 0.4714
    % 0.7858
    % 0.8232
    % 0.9799
    example.center = -1;
    example.radius = 1.9;
    
    example.num_eigs = 3; % 10
    example.is_symmetric = true;

    example.size = n^2;
end
    

function example = get_example_matieu(options)
% Matieu potential with a = -5, b = 5; and depth 50.

    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    example = struct();
    example.name = 'matieu';
    n = options.n;
    
    a = -25;
    b = 25;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);
    
    V = cos(xx)+cos(yy)-6*exp(-xx.^2-yy.^2);
    
    example.V = V;
    
    V = spdiags(V(:), 0, n^2, n^2); 
    
    if( output_full )
        A = kron( speye(n), T ) + kron( T, speye(n) );
        example.A = -A + V;
        example.B = speye( n^2 );
    end

    if( output_factored )
        if( options.is_potential_separate )
            example.M = speye( n );
            example.K = -T;

            % Cannot put the potentital only into VL, VR.
            % example.VL = sparse( n, n ); 
            % example.VR = sparse( n, n ); 

            example.Vleft{1} = speye( n );
            example.Vright{1} = spdiags( cos(x'), 0, n, n);

            example.Vleft{2} = spdiags( cos(x'), 0, n, n);
            example.Vright{2} = speye( n );

            example.Vleft{3} = sqrt(6)*spdiags( exp(-(x').^2), 0, n, n);
            example.Vright{3} = -sqrt(6)*spdiags( exp(-(y').^2), 0, n, n);
        else
            % warning( 'Part of the potential is in the K matrix.' );

            example.M = speye( n );
            example.K = -T + spdiags( cos(x'), 0, n, n);
            example.VL = sqrt(6)*spdiags( exp(-(x').^2), 0, n, n);
            example.VR = -sqrt(6)*spdiags( exp(-(y').^2), 0, n, n);

            example.Vleft{1} = example.VL;
            example.Vright{1} = example.VR;
        end
    end

    % [V,D] = eigs(example.A,30,'smallestreal');
    % diag(D)
    % pause
    % 5.0845e+00
    % 1.2361e+01
    % 1.3085e+01
    % 2.0691e+01
    % 2.4860e+01
    % Second gap
    
    % There is only 1 eigval inside the contour for n=100: -0.5675
    % (one to the left is -0.7041, one to the right is 0.2078)
    example.center = -2.39514e-01;
    example.radius = 4.55824e-01-1e-2;
    
    % Third gap
    % example.center = 7.42533e-01;
    % example.radius = 1.72144e-01;
    
    example.num_eigs = 1;
    example.is_symmetric = true;

    example.size = n^2;
end

function example = get_example_matieu_shifted(options)
% Matieu potential with a = -5, b = 5; and depth 50.

    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    example = struct();
    example.name = 'matieu_shifted';
    % n = 300; 
    n = options.n;
    
    a = -25;
    b = 25;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);
    
    V = cos(xx)+cos(yy)-6*exp(-xx.^2-yy.^2);
    
    example.V = V;
    
    % Low rank representation
    %[U1,S1,V1] = svd(V);
    %u1 = U1(:,1); v1 = V1(:,1)*S1(1,1);
    %V = kron(spdiags(u1,0,n,n),spdiags(v1,0,n,n));
        
    V = spdiags(V(:), 0, n^2, n^2); 
    
    if( output_full )
        A = kron( speye(n), T ) + kron( T, speye(n) );
        example.A = -A + V;
        example.B = speye( n^2 );
    end

    if( output_factored )
        if( options.is_potential_separate )
            example.M = speye( n );
            example.K = -T+0.1*eye(n);

            % Cannot put the potentital only into VL, VR.
            % example.VL = sparse( n, n ); 
            % example.VR = sparse( n, n ); 

            example.Vleft{1} = speye( n );
            example.Vright{1} = spdiags( cos(x'), 0, n, n);

            example.Vleft{2} = spdiags( cos(x'), 0, n, n);
            example.Vright{2} = speye( n );

            example.Vleft{3} = sqrt(6)*spdiags( exp(-(x').^2), 0, n, n);
            example.Vright{3} = -sqrt(6)*spdiags( exp(-(y').^2), 0, n, n);
        else
            % warning( 'Part of the potential is in the K matrix.' );

            example.M = speye( n );
            example.K = -T + spdiags( cos(x'), 0, n, n)+0.1*speye(n);
            example.VL = sqrt(6)*spdiags( exp(-(x').^2), 0, n, n);
            example.VR = -sqrt(6)*spdiags( exp(-(y').^2), 0, n, n);

            example.Vleft{1} = example.VL;
            example.Vright{1} = example.VR;
        end
    end

    % [V,D] = eigs(example.A,30,'smallestreal');
    % diag(D)
    % pause
    % 5.0845e+00
    % 1.2361e+01
    % 1.3085e+01
    % 2.0691e+01
    % 2.4860e+01
    % Second gap
    
    % There is only 1 eigval inside the contour for n=100: -0.5675
    % (one to the left is -0.7041, one to the right is 0.2078)
    example.center = -2.39514e-01;
    example.radius = 4.55824e-01-1e-2;
    
    % Third gap
    % example.center = 7.42533e-01;
    % example.radius = 1.72144e-01;
    
    % CAVEAT Had to increase the oversampling as compared to the full
    % representation of thepotential!
    
    % example.num_eigs = 10;
    example.num_eigs = 1;
    example.is_symmetric = true;

    example.size = n^2;
end


function example = get_example_double_well(options)
% Double well potential.

    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    assert( ~output_factored, 'Example double-well cannot be generated in factored form. Call generate_example with option output_mode=''full''.' );

    example = struct();
    example.name = 'double-well';
    n = options.n;

    a = -1;
    b = 1;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    A = kron( speye(n), T ) + kron( T, speye(n) );
    
    k = 1;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);
    
    X=spdiags(xx(:), 0, n^2, n^2); 
    Y=spdiags(yy(:), 0, n^2, n^2); 

    V = exp(-100 * ( (xx-1/4).^2 + (yy-1/4).^2 ) -3  ) + exp(-100 * ( (xx-3/4).^2 + (yy-3/4).^2 ) -3  );
    example.V = V;
    V=spdiags(V(:), 0, n^2, n^2); 
            
    example.A = -A + V;
    example.B = speye( n^2 );

    % [V,D] = eigs(example.A,5,'smallestreal');
    % diag(D)
    % pause
    % 4.9358
    % 12.3355
    % 12.3370
    % 19.7371
    % 24.6668
    example.center = 1.2336e+01;
    example.radius = 1.0000e+01;
    
    example.num_eigs = 4;
    example.is_symmetric = true;

    example.size = n^2;
end


function example = get_example_singular_one_over_sum_squares(options)
% Singular potential.

    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    assert( ~output_factored, 'Example singular-one-over-sum-squares cannot be generated in factored form. Call generate_example with option output_mode=''full''.' );

    example = struct();
    example.name = 'singular-one-over-sum-squares';
    n = options.n;

    a = -1;
    b = 1;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    A = kron( speye(n), T ) + kron( T, speye(n) );
    
    k = 1;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);
    
    X=spdiags(xx(:), 0, n^2, n^2); 
    Y=spdiags(yy(:), 0, n^2, n^2); 

    V = 1./sqrt(xx.^2 + yy.^2);
    example.V = V;
    V=spdiags(V(:), 0, n^2, n^2); 
                
    example.A = -A + V;
    example.B = speye( n^2 );

    % [V,D] = eigs(example.A,5,'smallestreal');
    % diag(D)
    % pause
    % 7.9069e+00
    % 1.4172e+01
    % 1.4172e+01
    % 2.1203e+01
    % 2.6322e+01
    example.center = 1.4000e+01;
    example.radius = 0.8000e+01;
    
    example.num_eigs = 4;
    example.is_symmetric = true;

    example.size = n^2;
end


function example = get_example_discontinuous(options)
    output_full = ( strcmp(options.output_mode, 'full') || strcmp(options.output_mode, 'both') );
    output_factored = ( strcmp(options.output_mode, 'factored') || strcmp(options.output_mode, 'both') );

    assert( ~output_factored, 'Example discontinuous cannot be generated in factored form. Call generate_example with option output_mode=''full''.' );

    example = struct();
    example.name = 'discontinuous';
    n = options.n;

    a = -1;
    b = 1;
    h = (b-a) / (n+1);
    
    % This assumes zero Dirchlet boundary conditions
    e = ones(n,1);
    T = spdiags([e -2*e e], -1:1, n, n) / h^2;
    A = kron( speye(n), T ) + kron( T, speye(n) );
    
    k = 1;
    
    x = linspace(a,b,n+2); x = x(2:end-1);
    y = linspace(a,b,n+2); y = y(2:end-1);
    [xx yy] = meshgrid(x,y);
    
    X = spdiags(xx(:), 0, n^2, n^2); 
    Y = spdiags(yy(:), 0, n^2, n^2); 

    V = -100 * ( sqrt(xx.^2+yy.^2)<3/4 ).*(xx.*yy>0) + (-5)*( sqrt(xx.^2+yy.^2)<3/4 ).*(xx.*yy<=0);
    example.V = V;
    V=spdiags(V(:), 0, n^2, n^2); 
                    
    example.A = -A + V;
    example.B = speye( n^2 );

    % [V,D] = eigs(example.A,5,'smallestreal');
    % diag(D)
    % pause
    % -7.4359e+01
    % -7.2842e+01
    % -4.3007e+01
    % -4.2501e+01
    % -3.7364e+01
    example.center = -5.8000e+01;
    example.radius =  1.8000e+01;
    
    example.num_eigs = 4;
    example.is_symmetric = true;

    example.size = n^2;
end
