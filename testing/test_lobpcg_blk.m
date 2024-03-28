% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Testing routine for the LOPCG method.

function [X, lam,res_vec,rank_vec,abs_error] = test_lobpcg_blk(namedparams)
    arguments
        namedparams.example = 'gauss';
        namedparams.n = 100;
        namedparams.random_vectors_type = 'rank-one';
        namedparams.verbosity = 2;
        namedparams.shift = 40;
        namedparams.tol = 0;
        namedparams.is_potential_separate = false;   
    end

    config = struct();

    % Available examples: 
    % 'sum-of-squares', 'sum-of-squares-minus-xy', 'deal-ii', 'trefethen-exp', 'double-well', 'singular-one-over-sum-squares', 
    % 'discontinuous', 'gauss', 'matieu'
    config.example_name = namedparams.example;

    config.problem_size = namedparams.n;  % Discretization size n (eigenvalue problem will be n^2 x n^2).
    config.is_potential_separate = namedparams.is_potential_separate;
    % Available random_vector_type-s:
    % 'gaussian';          % Ordinary random Gaussian vectors.
    % 'rank-one-multiply'; % Khatri-Rao product of random Gaussian vectors, multiplied out (not stored as Kronecker factors).
    % 'rank-one';          % Khatri-Rao product of random Gaussian vectors, kept in factored form. Sylvester solver must support this.
    config.random_vectors_type = namedparams.random_vectors_type;

    % Shift to make the problem positive definite.
    config.shift = namedparams.shift;
    
    % Tolerance for the eigenpair residual.
    config.tol = namedparams.tol;

    % Set verbosity to 0 for silence, and increase for more output.
    config.verbosity = namedparams.verbosity;
    logprintf( set_verbosity = config.verbosity );
       
    [X, lam,res_vec,rank_vec, example,abs_error] = run( config );
end


function [X, lam,res_vec,rank_vec, example,abs_error] = run( config )
    logprintf( 1, 'Preparing example ''%s''...', config.example_name );
    example = generate_example( config.example_name, n=config.problem_size );
    % example = generate_example( config.example_name, n=config.problem_size, output_mode='full' ); % Use this with examples that cannot be generated in the factored form
    logprintf( 1, 'done.\nEigenvalue problem size = %d.\n\n', example.size );

    logprintf( 1, 'Running LOBPCG ...\n' );
    logprintf( 2, '--------------------------------------------------------------\n' );
    
    t_start = tic;
    
    %load reference solution
    temp_ref=load(strcat(strcat('./Lobpcg_blk_plot/ref_sol/matlab_',example.name(1:3)),'.mat'))
    ref_sol=temp_ref.lam(1:1:4);
    clear temp_ref
    
    [X, lam,res_vec,rank_vec,abs_error] = lrlobpcg_blk( example, config, config.shift, ref_sol );
    t_end = toc(t_start);
    
    logprintf( 2, '--------------------------------------------------------------\n' );
    logprintf( 1, 'Finished in %.2f seconds.\n\n', t_end );
end
