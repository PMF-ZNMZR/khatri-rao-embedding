function [sA,sB] = pseudominmax(a,b)
% INPUT: a,b: sets of Ritz/Eigenvalues 
% OUTPUT: sA,sB: vectors with Sylvester-ADI shifts, s.t. product of
% spectralradii  is pseudo-minimized

% how many?
J=length(a);
L=length(b);
rr=zeros(J,L);

%evaluate Sylv-ADI rational function
for iA=1:J,
    for iB=1:L,
        tmp1=abs( (a-a(iA))./(a-b(iB)) );
        tmp2=abs( (b-b(iB))./(b-a(iA)) );
        rr(iA,iB)=max(tmp1)*max(tmp2);
    end
end
%find minimum
[val,idx] = min(rr);
[val, j] = min(val);
i = idx(j);

sA=a(i); sB=b(j);

if i~=J,  % move selected to the end
    tmp=a(J); a(J)=a(i); a(i)=tmp;
end
if j~=L,% move selected to the end
    tmp=b(L); b(L)=b(j); b(j)=tmp;
end

for ishift=2:J,
    kwa=J-ishift+1;
    kwb=L-ishift+1;
    rr=zeros(kwa,kwb);
    %evaluate Sylv-ADI rational function
    for iA=1:kwa,
        for iB=1:kwb,
            tmp1=abs( (a-a(iA))./(a-b(iB)) );
            tmp2=abs( (b-b(iB))./(b-a(iA)) );
            rr(iA,iB)=max(tmp1)*max(tmp2);
        end
    end
    %find minimum again
    [val,idx] = min(rr);
    [val, j] = min(val);
    i = idx(j);

    sA=[sA; a(i)]; sB=[sB; b(j)];
    if i~=kwa,  % move selected to the end
        tmp=a(kwa); a(kwa)=a(i); a(i)=tmp;
    end
    if j~=kwb,% move selected to the end
        tmp=b(kwb); b(kwb)=b(j); b(j)=tmp;
    end

end


