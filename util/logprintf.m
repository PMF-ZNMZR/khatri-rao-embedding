function logprintf( varargin, namedargs ) 
% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% A simple logger function allowing different levels of verbosity.

    arguments(Repeating)
        varargin
    end
    arguments
        namedargs.set_verbosity;
    end

    persistent verbosity;

    if( isempty(verbosity) )
        verbosity = 1;
    end
   
    if( isfield( namedargs, 'set_verbosity' ) )
        verbosity = namedargs.set_verbosity;
    end

    if( length(varargin) > 1 && verbosity >= varargin{1} )
        fprintf( varargin{2:end} );
    end
end
