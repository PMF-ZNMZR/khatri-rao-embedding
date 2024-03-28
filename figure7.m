function figure7()
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Generates Figure 7 from the manuscript.
    [~, ~,res_vec,rank_vec,abs_error]=test_lobpcg_blk_folded( example='matieu_shifted',verbosity=4, n=3000, shift=0); 

    % create plot
    figure();
    color_vec=["#0072BD","#D95319","#EDB120","#7E2F8E"];
    for count=1:3
        semilogy(1:size(res_vec,1),res_vec(:,count),color=color_vec(count),LineWidth=2,LineStyle="-")
        hold on;
    end
    set(gca,'YColor','k');
    xlabel('Iterations')
    ylabel('Residual')
    yyaxis right
    plot(1:size(rank_vec,2),rank_vec',"blue",Marker="o")
    xlim([1 size(res_vec,1)])
    ylabel('Rank')
    set(gca,'YColor','b');
    legend("residual #1","residual #2","residual #3","residual #4","rank of X")
    
    
    figure();
    for count=1:3
        semilogy(1:size(abs_error,1),abs_error(:,count),color=color_vec(count),LineWidth=2,LineStyle="-")
        hold on;
    end
    set(gca,'YColor','k');
    xlabel('Iterations')
    ylabel('Absolute error')    
    legend("eigenvalue #1","eigenvalue #2","eigenvalue #3","eigenvalue #4")
end
