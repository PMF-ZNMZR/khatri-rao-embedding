function figure3()
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Generates Figure 3 from the manuscript.
    
    figure3_left();
    figure3_right();
end


function figure3_left()
    test_contour( example='sum-of-squares-minus-xy', n=300, just_analyze_true_solution = true );
end


function figure3_right()
    test_sylvester( example='sum-of-squares-minus-xy', n=300, just_compare_rank1_vs_gaussian = true );
end
