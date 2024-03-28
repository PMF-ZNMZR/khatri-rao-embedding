% This file is part of the software library that reproduces 
% numerical experiments from the manuscript
%   Zvonimir Bujanović, Luka Grubišić, Daniel Kressner, Hei Yin Lam: 
%   "Subspace embedding with random Khatri-Rao products and its application to eigensolvers".
%
% Generates Figure 1 from the manuscript.     
function figure1()
    figure_left();
    figure_right();
end


function figure_left()
    rnd_state = rng(); rng(12345);
    eps=5;
    target_failure_prob=1/50;
    k_max=20;
    k_min=4;
    ntests = 1000;
    nsqrt = 20;
    gauss_ell=[];
    kr_ell=[];

    for k= k_min:k_max
        flag_ell=0;
        n = nsqrt*nsqrt;
        %create case 1
        V = randn( n, k );
        [V, ~] = qr( V, 0 );

        ell=k;
        while(flag_ell==0)
            gauss=zeros(ntests,1);
            for i=1:ntests
                %Gaussian embedding
                OM = randn(n, ell) / sqrt(ell);           
                 s = svd( V'*OM );

                gauss(i,1) = 1 ./ s(end);
            end
            
            if sum(gauss(:,1) > eps)./ntests<=target_failure_prob;
                flag_ell=1;
                gauss_ell=[gauss_ell,ell]
            end
            ell=ell+1;
        end

         ell=k;
         flag_ell=0;
        while(flag_ell==0)
            kr=zeros(ntests,1);
            for i=1:ntests
                %KR embedding
                OM = zeros(n, ell);
                OM_L = randn( nsqrt, ell ); OM_R = randn( nsqrt, ell );
                
                for j = 1 : ell 
                    OM(:, j) = kron( OM_L(:, j), OM_R(:, j) );
                end
                OM = OM / sqrt(ell);
                s = svd( V'*OM );

                kr(i,1) = 1 ./ s(end);
            end
            
            if sum(kr(:,1) > eps)./ntests<=target_failure_prob;
                flag_ell=1;
                kr_ell=[kr_ell,ell]
            end
            ell=ell+1;
        end
    end
    % create plot
    figure();
    bar([k_min:k_max],kr_ell,0.9,FaceColor="#0072BD",FaceAlpha=0.3)
    hold on
    bar([k_min:k_max],gauss_ell,0.9,FaceColor="#D95319",FaceAlpha=0.3)

    xlabel( ' $k$', 'interpreter', 'latex' );
    ylabel( '$\ell$', 'interpreter', 'latex' );
    legend('Khatri-Rao','Gaussian');
end

function figure_right()
    rnd_state = rng(); rng(12345);
    eps=5;
    target_failure_prob=1/50;
    k_max=20;
    k_min=4;
    ntests = 1000;
    nsqrt = 20;
    gauss_ell=[];
    kr_ell=[];
    for k= k_min:k_max
        flag_ell=0;
        
        %create case 2
        n = nsqrt*nsqrt;
        V = zeros( n, k );
        VL = randn(nsqrt, 1);
        VR=randn(nsqrt, nsqrt);
        [VR,~]=qr(VR);

        for i = 1 : k
        V(:, i) = kron( VL, VR(:,i));
        end
        V=V./(norm(V));

        ell=k
        while(flag_ell==0)
            gauss=zeros(ntests,1);
            for i=1:ntests
                % Gaussian embedding
                OM = randn(n, ell) / sqrt(ell);           
                s = svd( V'*OM );

                gauss(i,1) = 1 ./ s(end);
            end
            
            if sum(gauss(:,1) > eps)./ntests<=target_failure_prob;
                flag_ell=1;
                gauss_ell=[gauss_ell,ell]
            end
            ell=ell+1;
        end

         ell=k;
         flag_ell=0;
        while(flag_ell==0)
            %KR-embedding
            kr=zeros(ntests,1);
            for i=1:ntests
                OM = zeros(n, ell);
                OM_L = randn( nsqrt, ell ); OM_R = randn( nsqrt, ell );
                
                for j = 1 : ell 
                    OM(:, j) = kron( OM_L(:, j), OM_R(:, j) );
                end
                OM = OM / sqrt(ell);
                s = svd( V'*OM );

                kr(i,1) = 1 ./ s(end);
            end
            
            if sum(kr(:,1) > eps)./ntests<=target_failure_prob;
                flag_ell=1;
                kr_ell=[kr_ell,ell]
            end
            ell=ell+1;
        end
    end
    
    % create plot
    figure();
    bar([k_min:k_max],kr_ell,0.9,FaceColor="#0072BD",FaceAlpha=0.3)
    hold on
    bar([k_min:k_max],gauss_ell,0.9,FaceColor="#D95319",FaceAlpha=0.3)

    xlabel( ' $k$', 'interpreter', 'latex' );
    ylabel( '$\ell$', 'interpreter', 'latex' );
    legend('Khatri-Rao','Gaussian');
end

