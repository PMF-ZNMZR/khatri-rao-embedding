function set_path()
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Sets the path for the library.

    [filepath,~,~] = fileparts(mfilename('fullpath'));

    addpath( [filepath] ); 
    addpath( [filepath, '/util'] );
    addpath( [filepath, '/methods'] );
    addpath( [filepath, '/testing'] );
    
    % External libraries. Please install Tensor Toolbox v3.5 to the folder stated below. 
    addpath( [filepath, '/external/sylvester_ADI'] );
    addpath( [filepath, '/external/tensor_toolbox-v3.5'] );
end
