function figure6()
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Generates Figure 6 from the manuscript.
    example = generate_example( 'matieu', n=100 );
    e1=eigs((example.A+0.2*speye(10000))^2,110,'smallestreal');
    e2=eigs((example.A),100,'smallestreal');
    scatter(e1(1:100),zeros(100),Marker="o",MarkerEdgeColor="#EDB120",LineWidth=1.5)
    hold on
    scatter(e2,zeros(100),Marker="x",MarkerEdgeColor="#7E2F8E",LineWidth=1.5)
    xlim([-0.8 0.4])
    ylim([-0.2 0.2])
    grid on
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 0.35, 0.3]);
end
