function figure2()
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Generates Figure 2 from the manuscript.
    
    figure2_left();
    figure2_right();
end


function figure2_left()
    n = 400;
    k = 8;
    ntests = 1000;
    
    % U = random Gaussian n x k matrix, orthogonalized.
    rnd_state = rng(); rng(12345); 
    U = randn( n, k );
    [U, ~] = qr( U, 0 );
    rng(rnd_state);

    generate_figure( U, ntests );
    % save_picture( 'figure2_left.pdf' );
end


function figure2_right()
    nsqrt = 20;
    k = 8;
    ntests = 1000;

    n = nsqrt*nsqrt;

    % U = [u kron v1, u kron v2, ..., u kron vk].
    % V = random Gaussian nsqrt x k matrix, orthogonalized.
    % u = random Gaussian vector of size nsqrt, normalized.
    rnd_state = rng(); rng(12345); 
    V = randn( nsqrt, k );
    [V, ~] = qr( V, 0 );
    rng(rnd_state);

    U = zeros( n, k );
    u = randn(nsqrt, 1); u = u / norm(u);
    for i = 1 : k
        U(:, i) = kron( u, V(:, i) );
    end

    generate_figure( U, ntests );
    % save_picture( 'figure2_right.pdf' );
end


function generate_figure( U, ntests )
    [n, k] = size( U );
    nsqrt = sqrt( n );

    span_ell = k:(k+20);
    n_ell = length( span_ell );

    gauss = zeros( ntests, n_ell );
    kr = zeros( ntests, n_ell );
    for iell = 1:length(span_ell)
        ell = span_ell(iell);

        fprintf( '%d ', ell );

        for t = 1:ntests
            % Gaussian random matrix OM.
            OM = randn(n, ell) / sqrt(ell);
            s = svd( U'*OM );
            gauss(t, iell) = 1 ./ s(end);

            % Khatri-Rao random matrix OM.
            OM = zeros(n, ell);
            OM_L = randn( nsqrt, ell ); OM_R = randn( nsqrt, ell );

            for j = 1 : ell 
                OM(:, j) = kron( OM_L(:, j), OM_R(:, j) );
            end
            OM = OM / sqrt(ell);
            s = svd( U'*OM );

            kr(t, iell) = 1 ./ s(end);
        end

        gauss(:, iell) = sort( gauss(:, iell) );
        kr(:, iell) = sort( kr(:, iell) ); 
    end

    fprintf( '\n' );

    xlabels = span_ell;

    figure();
    semilogy( xlabels, gauss(end, :), '-', 'Color', [1, 0.7, 0.7], 'LineWidth',0.5, 'DisplayName', 'Gaussian, worst' );
    hold on;
    semilogy( xlabels, kr(end, :), '-', 'Color', [0.7, 0.7, 1], 'LineWidth', 0.5, 'DisplayName', 'Khatri-Rao, worst' );
    i95 = 0.95*ntests;
    semilogy( xlabels, gauss(i95, :), 'r-', 'LineWidth', 1.0, 'DisplayName', 'Gaussian, 95%' );
    hold on;
    semilogy( xlabels, kr(i95, :), 'b-', 'LineWidth', 1.0, 'DisplayName', 'Khatri-Rao, 95%' );
    imed = 0.5*ntests;
    semilogy( xlabels, gauss(imed, :), 'r:', 'LineWidth', 0.5, 'DisplayName', 'Gaussian, median' );
    hold on;
    semilogy( xlabels, kr(imed, :), 'b:', 'LineWidth', 0.5, 'DisplayName', 'Khatri-Rao, median' );

    ylim( [1e0, 1e8] );
    xlim( xlabels([1, end]) );

    xlabel( '$\ell$', 'interpreter', 'latex' );
    ylabel( '$\|(\Omega^T U)^\dag\|$', 'interpreter', 'latex' );
    legend( 'show' );
end
